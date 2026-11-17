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
  golden_faces.csv    notebook methods only: the plan's golden-zone faces (60 ft rule)
  manifest.json       config hash, git SHA, seed, input hashes, window, operators,
                      dispatch_legs, drop_off_mode, workload counts, leg-model check

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
from reslot.travel.artifacts import load_travel
from reslot.utils.manifest import write_manifest

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
if "notebook_plan" in x:
    rs = x["notebook_plan"]
    r = rs["workbook_replay"]
    print(f"\n{rs['label']} plan: {rs['plan']['swaps']} swaps"
          f"{' (blocked by backtest guard)' if rs['plan'].get('blocked_by_backtest_guard') else ''}, "
          f"move cost ${rs['move_cost_usd']:,.2f}")
    print(f"  {r['what']}: driving cut {r['walking_cut_pct']}%, gross ${r['gross_usd']:,.0f}, "
          f"net ${r['net_usd']:,.0f} on {r['tours']:,} tours")
    print(f"  not enforced: {'; '.join(rs['known_gaps'])}")
used = x["planning_inputs"]["lookahead_exceptions_used"]
if used:
    print(f"\nlookahead exceptions used (config lookahead_exceptions): {', '.join(used)}")
print(f"\nNOT findings: {x['not_findings']['note']}")
if manifest["git_dirty"]:
    print("NOTE working tree is dirty: this run is not citable")
print(f"wrote {run_dir}/")
