"""Print the reference operator count for the configured run window.

Run:  python scripts/operators_per_hour.py [configs/sim.yaml]

Median, over clock hours with at least one case pick (scope: WZPCK / CS picks),
of distinct operators (USER_NAME) -- the definition behind config `operators`
(decisions 2026-11-02). Reads data.picks_path over `window` only. Prints the
numbers; does not change the config.

Writes <paths.out>/operators_<start>_<end>/manifest.json (git SHA, config hash,
input hash, the counts).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reslot.config import load_config
from reslot.io.wms_picks import load_wms_picks
from reslot.sim.engine import window_bounds
from reslot.sim.operators import operators_per_hour
from reslot.utils.manifest import write_manifest

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/sim.yaml")
start, end = window_bounds(cfg)
path = Path(cfg["data"]["picks_path"])
if not path.exists():
    raise SystemExit(f"{path} not found")
events = load_wms_picks(path, cfg["data"]["ts_format"], window=(start, end))
r = operators_per_hour(events, cfg["scope"], start, end)

w = cfg["window"]
out = Path(cfg["paths"]["out"]) / f"operators_{w['start'].replace('-', '')}_{w['end'].replace('-', '')}"
out.mkdir(parents=True, exist_ok=True)
m = write_manifest(out, cfg, inputs=[path], extra={"operators_per_hour": r})

print(f"{w['start']} .. {w['end']}: {r['case_picks']:,} case picks over {r['hours_with_case_picks']} clock hours")
print(f"distinct case-pick operators per clock hour: median {r['median']:g}, p90 {r['p90']:g}, max {r['max']}")
print(f"distinct in window {r['distinct_operators_in_window']}, per day (median) {r['distinct_operators_per_day_median']:g}")
print(f"config operators is {cfg['operators']} (set from Oct 5-11); this script does not change it")
if m["git_dirty"]:
    print("NOTE working tree is dirty: this run is not citable")
print(f"wrote {out}/")
