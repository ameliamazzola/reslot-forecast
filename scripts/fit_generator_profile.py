"""Fit the order generator's profile on the planning window only.

Run:  python scripts/fit_generator_profile.py [configs/sim.yaml]

Reads racksmart.planning_picks over racksmart.planning_window (Sep 12 - Oct 11),
after the same export overlap check planning uses (racksmart.source_checks; it
reads nothing at or after the run start), deduplicated across exports. Scope:
WZPCK / CS case picks, VMS and discontinued vitamin-brand SKUs excluded
(generator.exclude: the affinity pipeline's filter). Writes generator.profile_path and a manifest beside it:

  orders_per_day, lines-per-order and cases-per-line histograms, hour-of-day and
  day-of-week shares of orders, SKU rates (lines per day), clusters (Louvain on
  order co-occurrence, config generator.clusters), and face_items (the last item
  picked at each face before the run start, for generator.sku_to_face: real).

Definitions: src/reslot/sim/generator.py.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from reslot.config import load_config
from reslot.sim import generator as G
from reslot.sim import sources as S
from reslot.sim.engine import window_bounds
from reslot.sim.racksmart import face_items
from reslot.utils.manifest import write_manifest

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/sim.yaml")
rc, scope, gcfg = cfg["racksmart"], cfg["scope"], cfg["generator"]
start, end = window_bounds({"window": rc["planning_window"]})
run_start, _ = window_bounds(cfg)
if end > run_start:
    raise SystemExit(f"planning window ends {end}, after the run start {run_start}")

try:
    checks = S.source_check(rc.get("source_checks"), rc["planning_ts_format"], scope, latest_end=end)
except ValueError as exc:
    raise SystemExit(f"not fitting: {exc}")
events, files = S.read_picks(rc["planning_picks"], rc["planning_ts_format"], (start, end))
events = S.dedup(events)
profile = G.fit_profile(events, scope, start, end, gcfg["clusters"], exclude=gcfg.get("exclude"))
items = face_items(events, scope, end)
profile["face_items"] = {str(k): str(v) for k, v in items.items()}
profile["face_items_rule"] = "last in-scope case pick at the face before the run start (planning picks only)"

out = Path(gcfg["profile_path"])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(profile, indent=1), encoding="utf-8")
inputs = sorted(set(S.entry_paths(rc["planning_picks"]) + S.check_files(rc.get("source_checks"))))
summary = {k: profile[k] for k in ("window", "days", "orders", "orders_per_day", "n_clusters", "skus_with_cluster")}
summary["skus"] = len(profile["sku_rates"])
manifest = write_manifest(out.parent, cfg, inputs=inputs,
                          extra={"profile": str(out), "profile_summary": summary, "planning_rows": files,
                                 "source_checks": checks, "definitions": G.DEFINITIONS,
                                 "lookahead": f"reads [{start}, {end}) only; run start {run_start}"})

h = profile["lines_per_order_hist"]
n = sum(h.values())
print(f"profile {start.date()} .. {(end - pd.Timedelta(days=1)).date()}: {profile['orders']:,} orders "
      f"({profile['orders_per_day']:.1f}/day), {summary['skus']:,} SKUs, {profile['n_clusters']} clusters "
      f"({profile['skus_with_cluster']} SKUs)")
print(f"lines per order: 1-2 lines {sum(v for k, v in h.items() if k <= 2) / n:.1%}, "
      f"max {max(h)}; cases per line: max {max(profile['cases_per_line_hist'])}")
print(f"faces with a last item: {len(profile['face_items']):,}")
x = profile["exclusion"]
print(f"excluded {x.get('excluded_skus', 0)} SKUs (VMS {x.get('vms_skus', 0)}, vitamin brands "
      f"{x.get('vitamin_brand_skus', 0)}): {x.get('lines_dropped', 0):,} lines, {x.get('orders_dropped', 0):,} orders")
if x.get("vms_skus_without_description"):
    print(f"NOTE {x['vms_skus_without_description']} VMS SKUs have no description: the ARRID keep rule "
          "could not see them, so they are excluded (as in the affinity pipeline)")
if manifest["git_dirty"]:
    print("NOTE working tree is dirty: this run is not citable")
print(f"wrote {out} and {out.parent / 'manifest.json'}")
