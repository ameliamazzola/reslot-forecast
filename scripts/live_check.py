"""Live-vs-replay check: feet and seconds per pick, generated vs real tours.

Run:  python scripts/live_check.py [configs/sim.yaml] [--hours 8] [--policy static] [--seed N]
                                   [--layouts real synthetic] [--mode default|empirical|resample]

--mode overrides generator.mode for this run only (default: the config's); the
mode and its description label every live row and go into the manifest.

Measurement boundary (decisions 2026-11-20): arrivals stop at --hours and both
sides drain. A tour counts, whole, if and only if it was released in
[window start, window start + --hours); every counted tour runs to completion,
however late it finishes; nothing released at or after the cutoff counts.

  live    for each layout, the live engine (the same LiveRun run_live.py uses, on
          a LiveEnvironment at an unthrottled speed) with the configured routing
          rule, admitting arrivals before the cutoff and running until every
          admitted tour is done (a run that doesn't drain is refused)
  replay  the static replay run's tours.csv in paths.out (every tour already run
          to completion), restricted to tours released before the cutoff

Units. Replay counts scan rows: one SKU on one order can be scanned in several
rows at one stop (0-ft legs), while a generated order has one pick per SKU line.
So feet and seconds are reported per

  SKU line     distinct SKU per tour                                   PRIMARY
  stop visit   consecutive picks at one stop, in route order
  tour
  pick row     one scan row (the unit used before; not comparable across sides)

Feet = the engine's pick-to-pick leg feet (tours.pick_feet), dispatch legs
excluded; seconds = operator busy time (tours.busy_s). Both sides come from the
engine's tours table (live: the run's own; replay: the static run's tours.csv,
which must carry n_lines / n_stop_visits -- rerun run_sim.py if it predates them).
Writes <paths.out>/live_check_<timestamp>/manifest.json.
"""
import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import pandas as pd

from reslot.config import load_config
from reslot.sim import engine as E
from reslot.sim import generator as G
from reslot.sim import live as L
from reslot.sim.policy import make_policy
from reslot.sim.racksmart import METHODS
from reslot.travel.artifacts import load_travel
from reslot.utils.manifest import write_manifest

UNTHROTTLED = 1e9
COUNTS = ["side", "routing", "tours", "lines", "stop_visits", "picks", "feet", "operator_hours", "drain_end_s"]
PER_UNIT = [f"{q}_per_{u}" for u, _, _ in L.UNITS for q in ("ft", "s")]

ap = argparse.ArgumentParser(description="Feet and seconds per pick: live (generated) vs replay (real).")
ap.add_argument("config", nargs="?", default="configs/sim.yaml")
ap.add_argument("--hours", type=float, default=8.0, help="arrival cutoff, sim hours from the window start")
ap.add_argument("--policy", default="static", choices=["static", *METHODS])
ap.add_argument("--seed", type=int)
ap.add_argument("--layouts", nargs="+", default=["real", "synthetic"], choices=["real", "synthetic"])
ap.add_argument("--mode", choices=list(G.MODES), help="override generator.mode for this run")
args = ap.parse_args()

cfg = load_config(args.config)
start, end = E.window_bounds(cfg)
arrival_cutoff_s = args.hours * 3600
if not math.isfinite(arrival_cutoff_s) or not 0 < arrival_cutoff_s <= (end - start).total_seconds():
    ap.error("--hours must be positive and within the configured simulation window")
arrival_end = start + pd.Timedelta(seconds=arrival_cutoff_s)
rule = L.boundary_rule(start, arrival_cutoff_s)
try:
    leg_model = E.make_leg_model(cfg)
except ValueError as exc:
    raise SystemExit(f"not running: {exc}")
net = load_travel(cfg["travel"]["dir"])
days = (end - start).days
gcfg = dict(cfg["generator"])
if args.seed is not None:
    gcfg["seed"] = args.seed
if args.mode:
    gcfg["mode"] = args.mode
mode = gcfg.get("mode", "empirical")
profile_path = Path(gcfg["profile_path"])
profile = json.loads(profile_path.read_text(encoding="utf-8"))
policy, move_cost, rs_files = make_policy("static"), None, []
if args.policy != "static":
    from reslot.sim import racksmart as RS
    policy, _, rs_files = RS.load_racksmart(cfg, net, method=args.policy)
    move_cost = policy.move_cost
E.check_planning_inputs(policy, start, cfg.get("lookahead_exceptions"))

rows = []
for layout in args.layouts:
    src = G.make_generator_source({**gcfg, "sku_to_face": layout}, cfg["scope"], net, profile, start, days,
                                  dispatch_stop=cfg["trip_ends"]["start_stop"])
    mode_label = src.label
    run = L.LiveRun(src, policy, net, leg_model, cfg["scope"], cfg["trip_ends"], cfg["operators"], UNTHROTTLED,
                    [UNTHROTTLED], move_cost=move_cost, arrival_cutoff_s=arrival_cutoff_s)
    run.start()
    run.wait()
    try:
        rows.append(L.live_window_row(run, f"live {mode}, layout {layout}", src.routing_rule))
    except ValueError as exc:
        raise SystemExit(f"not comparable: {exc}")

ref = Path(cfg["paths"]["out"]) / E.run_dir_name(cfg, "static", leg_model.name) / "tours.csv"
ref_inputs = []
if ref.exists():
    ref_inputs = [ref]
    try:
        rows.append(L.replay_window_row(pd.read_csv(ref), arrival_cutoff_s))
    except ValueError as exc:
        raise SystemExit(f"replay reference {ref}: {exc}")
table = pd.DataFrame(rows, columns=COUNTS + PER_UNIT)

stamp = datetime.now().strftime('%Y%m%dT%H%M%S')
out, k = Path(cfg["paths"]["out"]) / f"live_check_{stamp}", 1
while out.exists():                                   # two runs in one second never share a folder
    k += 1
    out = Path(cfg["paths"]["out"]) / f"live_check_{stamp}_{k}"
manifest = write_manifest(out, {**cfg, "generator": gcfg}, inputs=[profile_path, *ref_inputs, *rs_files],
                          extra={"mode": "live_check", "citable": False, "hours": args.hours, "policy": policy.name,
                                 "generator_mode": mode, "generator_mode_label": mode_label,
                                 "boundary_rule": rule,
                                 "units": {u: label for u, _, label in L.UNITS}, "primary_unit": "SKU line",
                                 "arrival_window": {"start": str(start), "end_exclusive": str(arrival_end),
                                                    "seconds": arrival_cutoff_s},
                                 "operators": cfg["operators"],
                                 "dispatch_legs": cfg["trip_ends"]["dispatch_legs"],
                                 "drop_off_mode": cfg["trip_ends"]["drop_off_mode"],
                                 "replay_reference": str(ref) if ref.exists() else None,
                                 "rows": table.astype(object).where(table.notna(), None).to_dict("records")})
print(f"Generator: {mode_label}")
print(f"Boundary: {rule}")
with pd.option_context("display.width", 220):
    print(table[COUNTS].round(2).to_string(index=False, na_rep="n/a"))
    per = table[["side"] + PER_UNIT].round(1)
    per.columns = ["side"] + [f"{q} per {u}{' (PRIMARY)' if u == L.PRIMARY_UNIT else ''}"
                              for u, _, _ in L.UNITS for q in ("ft", "s")]
    print()
    print(per.to_string(index=False, na_rep="n/a"))
print("PRIMARY unit: SKU line (distinct SKU per tour). Pick rows are not comparable across sides: replay scans "
      "one line in several rows, the generator emits one pick per line.")
print("feet: pick-to-pick, dispatch legs excluded; seconds: operator busy time; "
      "drain_end_s: last counted tour's completion, s from the window start")
if not ref.exists():
    print(f"no replay reference: {ref} not found (run scripts/run_sim.py --policy static)")
if manifest["git_dirty"]:
    print("NOTE working tree is dirty: this run is not citable")
print(f"wrote {out}/manifest.json")
