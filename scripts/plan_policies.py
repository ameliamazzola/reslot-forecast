"""Plan every notebook method and score each the notebook's way -- no SimPy timing.

Run:  python scripts/plan_policies.py [configs/sim.yaml]

Runs notebooks/RackSmart_final.ipynb's algorithm definitions unchanged (see
src/reslot/sim/racksmart.py): RackSmart through the notebook's own plan_month,
the comparison methods wired as the previous notebook's run_month wired them:

  racksmart         RackSmart (final)
  classic_abc       Classic ABC (no cost test)
  abc_cost_test     ABC with cost test
  family_grouping   Family grouping + cost test

All planned once on racksmart.planning_window (the month before Sep 21), on the
same tours, Inputs and cost model; each scored on `window` (Oct 5-11) with the
notebook's replay (feet x $/ft, $0.17/case). Needs no leg-time model.

Writes data/processed/sim/notebook_plans_<start>_<end>/ (config paths.out):
  comparison.csv            one row per method (plus Do nothing), check_month's columns
  <method>_swaps.csv        the notebook's swap_list per method
  manifest.json             config hash, git SHA, input hashes, every plan and its scoring
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from reslot.config import load_config
from reslot.io.wms_picks import load_wms_picks
from reslot.sim import racksmart as RS
from reslot.sim.engine import check_planning_inputs, window_bounds
from reslot.sim.workload import build_workload
from reslot.travel.artifacts import load_travel
from reslot.utils.manifest import write_manifest

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/sim.yaml")
net = load_travel(cfg["travel"]["dir"])
net.require_stops(cfg["trip_ends"]["start_stop"])
picks_path = cfg["data"]["picks_path"]
events = load_wms_picks(picks_path, cfg["data"]["ts_format"])

nb, inp, train, base, files = RS.prepare_planning(cfg, net)
policies = {"racksmart": RS.plan_policy(nb, inp, train), **RS.plan_comparison(nb, inp, train)}
start, end = window_bounds(cfg)
for pol in policies.values():                    # same inputs for every method; same check as engine.run
    pol.planning_inputs = base["planning_inputs"]
try:
    lookahead = check_planning_inputs(policies["racksmart"], start, cfg.get("lookahead_exceptions"))
except ValueError as exc:
    raise SystemExit(f"not planning: {exc}")
week = build_workload(events, net, cfg["scope"], cfg["tour"], start, end)

w = cfg["window"]
run_dir = Path(cfg["paths"]["out"]) / f"notebook_plans_{w['start'].replace('-', '')}_{w['end'].replace('-', '')}"
run_dir.mkdir(parents=True, exist_ok=True)
rows, plans = [{"method": "Do nothing", "swaps": 0, "walking_cut_pct": 0.0, "gross_usd": 0.0,
                "move_cost_usd": 0.0, "net_usd": 0.0}], {}
for key in RS.METHODS:
    pol = policies[key]
    replay = RS.workbook_replay(pol, week.tours)
    info = RS.method_info(pol, base)
    plans[key] = {k: info[k] for k in ("label", "plan", "moves", "move_cost_usd", "known_gaps")}
    plans[key]["workbook_replay"] = replay
    RS.swap_table(pol).to_csv(run_dir / f"{key}_swaps.csv", index=False)
    rows.append({"method": info["label"], "swaps": pol.plan["swaps"], "walking_cut_pct": replay["walking_cut_pct"],
                 "gross_usd": replay["gross_usd"], "move_cost_usd": replay["move_cost_usd"], "net_usd": replay["net_usd"]})
table = pd.DataFrame(rows)
table.to_csv(run_dir / "comparison.csv", index=False)
shared = {k: v for k, v in base.items()}
manifest = write_manifest(run_dir, cfg, inputs=[picks_path, net.sources["distance_table"],
                                                net.sources["crosswalk"], *files],
                          extra={"window": {"start": w["start"], "end": w["end"]},
                                 "trip_start": cfg["trip_ends"]["start_stop"],
                                 "scored_week_workload": {k: week.counts[k] for k in
                                                          ("picks_kept", "picks_dropped", "tours")},
                                 "planning": shared, "planning_inputs": lookahead, "plans": plans,
                                 "scoring": "the notebook's replay and cost model (feet and $), not the SimPy timing"})

print(f"notebook {base['notebook']['path']} (sha {base['notebook']['sha256_16']}); planned on "
      f"{base['planning_window']['start']}..{base['planning_window']['end']} "
      f"({base['inputs']['planning_tours']:,} tours), scored on {w['start']}..{w['end']} ({week.counts['tours']:,} tours)")
print(table.to_string(index=False))
print("scoring: the notebook's replay and cost model (feet and $), not the SimPy timing")
if manifest["git_dirty"]:
    print("NOTE working tree is dirty: this run is not citable")
print(f"wrote {run_dir}/")
