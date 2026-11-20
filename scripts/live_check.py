"""Live-vs-replay check: feet and seconds per pick, generated vs real tours.

Run:  python scripts/live_check.py [configs/sim.yaml] [--hours 8] [--policy static] [--seed N]
                                   [--layouts real synthetic]

For each layout, runs the live engine (the same LiveRun run_live.py uses, on a
LiveEnvironment at an unthrottled speed, so it takes seconds) for --hours of sim
time from the window start, with the configured routing rule; then prints
pick-to-pick feet per pick (dispatch legs excluded) and leg-model seconds per
pick next to replay's, read from the static replay run's tours.csv in paths.out
(whole window, and the tours released in the same first --hours).

The quantities are the same on both sides: feet = the engine's leg feet summed
(replay tours.pick_feet; live "feet so far"), seconds = operator busy time (replay
tours.busy_s; live operator-hours x 3600), both divided by picks (first picks
included). Writes <paths.out>/live_check_<timestamp>/manifest.json.
"""
import argparse
import json
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

ap = argparse.ArgumentParser(description="Feet and seconds per pick: live (generated) vs replay (real).")
ap.add_argument("config", nargs="?", default="configs/sim.yaml")
ap.add_argument("--hours", type=float, default=8.0)
ap.add_argument("--policy", default="static", choices=["static", *METHODS])
ap.add_argument("--seed", type=int)
ap.add_argument("--layouts", nargs="+", default=["real", "synthetic"], choices=["real", "synthetic"])
args = ap.parse_args()

cfg = load_config(args.config)
try:
    leg_model = E.make_leg_model(cfg)
except ValueError as exc:
    raise SystemExit(f"not running: {exc}")
net = load_travel(cfg["travel"]["dir"])
start, end = E.window_bounds(cfg)
days = (end - start).days
gcfg = dict(cfg["generator"])
if args.seed is not None:
    gcfg["seed"] = args.seed
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
    run = L.LiveRun(src, policy, net, leg_model, cfg["scope"], cfg["trip_ends"], cfg["operators"], UNTHROTTLED,
                    [UNTHROTTLED], until_s=args.hours * 3600, move_cost=move_cost)
    run.start()
    run.wait()
    t = run.totals()
    rows.append({"side": f"live, layout {layout}", "routing": src.routing_rule, "tours": t["orders_done"],
                 "picks": t["picks_done"], "feet": t["pick_feet"], "seconds": t["operator_hours"] * 3600})

ref = Path(cfg["paths"]["out"]) / E.run_dir_name(cfg, "static", leg_model.name) / "tours.csv"
ref_inputs = []
if ref.exists():
    tr = pd.read_csv(ref)
    ref_inputs = [ref]
    first = tr[tr["arrival_s"] < args.hours * 3600]
    for label, d in ((f"replay static, first {args.hours:g} h", first), ("replay static, whole window", tr)):
        rows.append({"side": label, "routing": "actual scan order", "tours": len(d), "picks": int(d["n_picks"].sum()),
                     "feet": float(d["pick_feet"].sum()), "seconds": float(d["busy_s"].sum())})
table = pd.DataFrame(rows)
table["ft_per_pick"] = table["feet"] / table["picks"]
table["s_per_pick"] = table["seconds"] / table["picks"]

out = Path(cfg["paths"]["out"]) / f"live_check_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
manifest = write_manifest(out, {**cfg, "generator": gcfg}, inputs=[profile_path, *ref_inputs, *rs_files],
                          extra={"mode": "live_check", "citable": False, "hours": args.hours, "policy": policy.name,
                                 "rows": table.to_dict("records")})
with pd.option_context("display.width", 160):
    print(table[["side", "routing", "tours", "picks", "feet", "ft_per_pick", "s_per_pick"]].round(1).to_string(index=False))
if not ref.exists():
    print(f"no replay reference: {ref} not found (run scripts/run_sim.py --policy static)")
print(f"wrote {out}/manifest.json")
