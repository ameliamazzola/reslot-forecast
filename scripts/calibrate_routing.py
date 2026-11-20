"""Choose the visit-order rule for generated tours against real tours.

Run:  python scripts/calibrate_routing.py [configs/sim.yaml]

Reads the planning window (racksmart.planning_window, Sep 12 - Oct 11) the way
planning does: racksmart.planning_picks after the export overlap check
(racksmart.source_checks; nothing at or after the run start), deduplicated, and
builds the real case-pick tours (order x operator, kept picks in scan order) on
the travel network. Each routing rule (src/reslot/sim/routing.py: draw,
work_path, aisle_serpentine, nearest_neighbour) re-orders every tour's picks;
pick-to-pick feet per pick (dispatch legs excluded, as replay's pick_feet) is
compared with the tours' actual scan order. The rule closest to actual is
chosen; generator.routing.rule: calibrated uses it.

Writes generator.routing.calibration_path (JSON) and a manifest beside it.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from reslot.config import load_config
from reslot.sim import routing as R
from reslot.sim import sources as S
from reslot.sim.engine import window_bounds
from reslot.sim.workload import build_workload
from reslot.travel.artifacts import load_travel
from reslot.utils.manifest import write_manifest

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/sim.yaml")
rc, scope, rt = cfg["racksmart"], cfg["scope"], cfg["generator"]["routing"]
start, end = window_bounds({"window": rc["planning_window"]})
run_start, _ = window_bounds(cfg)
if end > run_start:
    raise SystemExit(f"planning window ends {end}, after the run start {run_start}")
net = load_travel(cfg["travel"]["dir"])
dispatch = cfg["trip_ends"]["start_stop"]
net.require_stops(dispatch)

try:
    checks = S.source_check(rc.get("source_checks"), rc["planning_ts_format"], scope, latest_end=end)
except ValueError as exc:
    raise SystemExit(f"not calibrating: {exc}")
# as planning reads it: one day before the window so tours starting before it are seen, nothing at or after its end
events, files = S.read_picks(rc["planning_picks"], rc["planning_ts_format"], (start - pd.Timedelta(days=1), end))
events = S.dedup(events)
work = build_workload(events, net, scope, cfg["tour"], start, end)
router = R.Router(net, scope, dispatch, R.work_path_series(rt["location_master"]))
res = R.calibrate(work.tours, router)
res["window"] = [str(start), str(end)]
res["dispatch_stop"] = dispatch
res["workload"] = {k: work.counts[k] for k in ("picks_kept", "picks_dropped", "tours")}

out = Path(rt["calibration_path"])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(res, indent=1), encoding="utf-8")
inputs = sorted(set(S.entry_paths(rc["planning_picks"]) + S.check_files(rc.get("source_checks"))
                    + [rt["location_master"], net.sources["distance_table"], net.sources["crosswalk"]]))
manifest = write_manifest(out.parent, cfg, inputs=inputs,
                          extra={"routing_calibration": res, "source_checks": checks,
                                 "lookahead": f"reads [{start}, {end}) only; run start {run_start}"})

fp = res["feet_per_pick"]
print(f"{res['tours']:,} real tours, {res['picks']:,} picks, {start.date()} .. {end.date()} (exclusive)")
print(f"{'order':20s} {'ft/pick':>8s} {'diff':>8s} {'median rho':>11s}")
print(f"{'actual scan order':20s} {fp['actual_scan_order']:8.1f}")
for r in R.RULES:
    rho = res["median_spearman_with_scan_order"][r]
    print(f"{r:20s} {fp[r]:8.1f} {fp[r] - fp['actual_scan_order']:+8.1f} {'' if rho is None else f'{rho:11.3f}'}"
          f"{'   <- chosen' if r == res['chosen'] else ''}")
if manifest["git_dirty"]:
    print("NOTE working tree is dirty: this run is not citable")
print(f"wrote {out}")
