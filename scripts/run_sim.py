"""Replay one window of case-pick tours through the SimPy loop under one policy.

Run:  python scripts/run_sim.py [configs/sim.yaml] [--policy static|racksmart|classic_abc|abc_cost_test|family_grouping]
                                [--move-cases inventory_snapshot|pallet_fill]

Reads:
  data.picks_path                                   converted transactions export
  <travel.dir>/stop_distance_ft.csv                 from build_travel_network.py
  <travel.dir>/location_stop_crosswalk.csv          from build_travel_network.py

Writes data/processed/sim/sim_<policy>_<leg model>_<start>_<end>[_pallet_fill]/ (config paths.out;
the suffix marks a notebook method run with the snapshot-free move cost):
  tours.csv           one row per tour: wait, busy, scan span, pick feet, dispatch feet
  legs.csv            one row per simulated pick-to-pick leg, with observed seconds
  dropped_picks.csv   every in-window case pick that could not be placed, with its reason
  swaps.csv           notebook methods only: the plan's swaps (notebook swap_list)
  golden_faces.csv    notebook methods only: the plan's golden-zone faces (nearest 10.8% to dispatch)
  manifest.json       config hash, git SHA, seed, input hashes, window, operators,
                      dispatch_legs, drop_off_mode, workload counts, leg-model check,
                      headline (net labor savings vs static, payback days)

Headline (src/reslot/sim/headline.py): operator-hours saved vs static x W - move
cost, and payback days; W = config headline.labor_rate_usd_per_hour. The static
reference is the static run in the same output folder when it ran on the same
window, operators, leg model, scope, trip ends and input files (run_all.py runs
static first); otherwise static is simulated here on the same tours. The
notebook's feet scoring is reported as the secondary figure.

Refuses to run unless leg_time.model names a registered model that loads -- the
network_feet model loads only if its refit was adopted by the 2026-11-02 review.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reslot.config import load_config
from reslot.io.wms_picks import load_wms_picks
from reslot.sim import engine as E
from reslot.sim import headline as H
from reslot.sim.policy import make_policy
from reslot.travel.artifacts import load_travel
from reslot.utils.manifest import file_hash, write_manifest

ap = argparse.ArgumentParser(description="Replay one window of case-pick tours under one policy.")
ap.add_argument("config", nargs="?", default="configs/sim.yaml")
ap.add_argument("--policy", help="override the config's policy for this run (recorded in the manifest's config)")
ap.add_argument("--move-cases", choices=["inventory_snapshot", "pallet_fill"],
                help="override racksmart.move_cases for this run (recorded in the manifest's config)")
args = ap.parse_args()
cfg = load_config(args.config)
if args.policy:
    cfg["policy"] = args.policy
if args.move_cases:
    cfg["racksmart"]["move_cases"] = args.move_cases
try:
    leg_model = E.make_leg_model(cfg)
    rate, rate_source = H.labor_rate(cfg)
except ValueError as exc:
    raise SystemExit(f"not running: {exc}")

picks_path = Path(cfg["data"]["picks_path"])
if not picks_path.exists():
    raise SystemExit(f"{picks_path} not found")
net = load_travel(cfg["travel"]["dir"])
events = load_wms_picks(picks_path, cfg["data"]["ts_format"])

policy, move_cost, rs_files = None, None, []
from reslot.sim import racksmart as RS
if cfg["policy"] in RS.METHODS:                     # a notebook method (RackSmart, ABC, family grouping)
    policy, rs_info, rs_files = RS.load_racksmart(cfg, net, method=cfg["policy"])
    move_cost = policy.move_cost

out = E.run(cfg, events, net, leg_model, policy=policy, move_cost=move_cost)
x, res = out.extra, out.result
if policy is not None:
    x["notebook_plan"] = {**rs_info, "workbook_replay": RS.workbook_replay(policy, out.tours)}

# --- headline: net labor savings vs static ---------------------------------
days = H.window_days(cfg["window"])
hours = x["not_findings"]["busy_hours"]
if x["policy"] == "static":
    static_hours, ref = hours, {"from": "this run (static is the reference)"}
else:
    ref_dir = Path(cfg["paths"]["out"]) / E.run_dir_name(cfg, "static", x["leg_time_model"])
    sm = H.read_static(ref_dir)
    shared = [picks_path, net.sources["distance_table"], net.sources["crosswalk"], *getattr(leg_model, "files", [])]
    why = ["no static run"] if sm is None else H.static_mismatch(sm, x, {Path(p).name: file_hash(p) for p in shared})
    if not why:
        static_hours = float(sm["not_findings"]["busy_hours"])
        ref = {"from": str(ref_dir), "git_sha": sm.get("git_sha"), "git_dirty": sm.get("git_dirty")}
    else:                                             # same tours, same leg model, no plan
        st_out = E.run(cfg, events, net, leg_model, policy=make_policy("static"))
        static_hours = st_out.extra["not_findings"]["busy_hours"]
        ref = {"from": "simulated in this run", "why_not_the_static_folder": why}
hl = H.headline(static_hours, hours, x["moves"]["cost"], rate, days)
x["headline"] = {**hl, "labor_rate_source": rate_source, "static_reference": ref,
                 "definition": H.DEFINITION, "caveat": H.CAVEAT,
                 "secondary_notebook_net_usd": ((x.get("notebook_plan") or {}).get("workbook_replay") or {}).get("net_usd")}

w = cfg["window"]
run_dir = Path(cfg["paths"]["out"]) / E.run_dir_name(cfg, x["policy"], x["leg_time_model"])
run_dir.mkdir(parents=True, exist_ok=True)
res.tours.to_csv(run_dir / "tours.csv", index=False)
res.legs.to_csv(run_dir / "legs.csv", index=False)
out.dropped.to_csv(run_dir / "dropped_picks.csv", index=False)
if policy is not None:
    RS.swap_table(policy).to_csv(run_dir / "swaps.csv", index=False)
    RS.golden_table(policy).to_csv(run_dir / "golden_faces.csv", index=False)
inputs = [picks_path, net.sources["distance_table"], net.sources["crosswalk"], *rs_files,
          *getattr(leg_model, "files", [])]
manifest = write_manifest(run_dir, cfg, inputs=inputs, extra=x)

# --- report ---------------------------------------------------------------
wl, chk = x["workload"], x["leg_model_check"]
print(f"{x['policy']} | leg model {x['leg_time_model']} | {w['start']} .. {w['end']} | "
      f"{x['operators']} operators | dispatch_legs {x['dispatch_legs']}, drop_off {x['drop_off_mode']}")
print(f"{wl['picks_in_window_tours']:,} case picks in window tours -> {wl['picks_kept']:,} simulated in "
      f"{wl['tours']:,} tours; dropped {wl['picks_dropped']:,} ({wl['dropped_share']:.1%})")
for reason, n in wl["dropped_by_reason"].items():
    print(f"  dropped {reason:28s} {n:7,}")
print(f"tours: {wl['tours_with_dropped_picks']:,} with dropped picks, {wl['tours_single_pick']:,} single-pick, "
      f"{wl['tours_overlapping_same_user']:,} overlapping the same operator's previous tour")
p, s = chk["primary_pick_to_pick_legs"], chk["secondary_first_to_last_scan"]
print(f"\n{chk['what']} ({chk['leg_time_model']}):")
if p["n_legs"]:
    print(f"  primary   {p['n_legs']:,} legs: MAE {p['mae_s']:.1f} s, median sim {p['median_sim_s']:.0f} s "
          f"vs obs {p['median_obs_s']:.0f} s, mean error {p['mean_error_s']:+.1f} s")
if s["n_tours"]:
    print(f"  secondary {s['n_tours']:,} complete tours: median abs pct error {s['median_abs_pct_error']:.1%}, "
          f"total sim/obs {s['total_sim_over_obs']:.2f}")
print(f"  [{chk['source']}] {chk['caveat']}")
h = x["headline"]
pb = ("no moves" if h["move_cost_usd"] <= 0 else "never (saves no hours)" if h["payback_days"] is None
      else f"{h['payback_days']:.1f} days" + ("" if h["pays_back_in_window"] else f", longer than the {days}-day window"))
print(f"\nHEADLINE net labor savings vs static: ${h['net_usd']:,.0f} over {days} days "
      f"({h['operator_hours_saved']:+,.1f} operator-hours x ${rate:.2f}/h = ${h['labor_saved_usd']:,.0f}, "
      f"move cost ${h['move_cost_usd']:,.0f}); payback {pb}")
print(f"  W: {rate_source}. Static reference: {h['static_reference']['from']}. {H.CAVEAT}")
if "notebook_plan" in x:
    rs = x["notebook_plan"]
    r = rs["workbook_replay"]
    print(f"\n{rs['label']} plan: {rs['plan']['swaps']} swaps"
          f"{' (blocked by backtest guard)' if rs['plan'].get('blocked_by_backtest_guard') else ''}, "
          f"move cost ${rs['move_cost_usd']:,.2f}")
    print(f"  secondary -- {r['what']}: driving cut {r['walking_cut_pct']}%, gross ${r['gross_usd']:,.0f}, "
          f"net ${r['net_usd']:,.0f} on {r['tours']:,} tours")
    print(f"  not enforced: {'; '.join(rs['known_gaps'])}")
used = x["planning_inputs"]["lookahead_exceptions_used"]
if used:
    print(f"\nlookahead exceptions used (config lookahead_exceptions): {', '.join(used)}")
print(f"\nNOT findings: {x['not_findings']['note']}")
if manifest["git_dirty"]:
    print("NOTE working tree is dirty: this run is not citable")
print(f"wrote {run_dir}/")
