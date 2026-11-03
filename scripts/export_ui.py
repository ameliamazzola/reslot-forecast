"""Bundle sim runs + the travel network into one JSON file for ui/index.html.

Run:  python scripts/export_ui.py RUN_DIR [RUN_DIR ...] [--travel DIR] [--out FILE]

  RUN_DIR   runs/sim_<policy>_<model>_<start>_<end>/ from scripts/run_sim.py,
            all over the same window and network (e.g. static and racksmart)
  --travel  network outputs (default: travel.dir in configs/sim.yaml)
  --out     default: data/processed/ui/bundle_<start>_<end>.json
  --affinity  default: data/processed/affinity_graph.json (from src/affinity/affinity_graph.py)

The bundle is derived from C&D data, so it goes under data/processed/
(gitignored). Open ui/index.html and choose the file.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from reslot.config import load_config
from reslot.sim.export import build_bundle
from reslot.travel.artifacts import CROSSWALK_FILE, DISTANCE_FILE, load_travel
from reslot.utils.manifest import file_hash

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("runs", nargs="+")
ap.add_argument("--travel", default=None)
ap.add_argument("--out", default=None)
ap.add_argument("--affinity", default="data/processed/affinity_graph.json",
                help="affinity graph from src/affinity/affinity_graph.py (skipped if missing)")
a = ap.parse_args()

travel = Path(a.travel or load_config("configs/sim.yaml")["travel"]["dir"])
net = load_travel(travel)
stops = pd.read_csv(travel / "stops.csv", dtype={"stop_id": str, "label": str})
hashes = {f: file_hash(travel / f) for f in (DISTANCE_FILE, CROSSWALK_FILE)}

aff_path = Path(a.affinity)
affinity = json.loads(aff_path.read_text(encoding="utf-8")) if aff_path.exists() else None
if affinity is None:
    print(f"no affinity graph at {aff_path}: run src/affinity/affinity_graph.py to add it")
bundle = build_bundle(a.runs, net, stops, datetime.now(timezone.utc).isoformat(), hashes, affinity)
w = bundle["window"]
out = Path(a.out or f"data/processed/ui/bundle_{w['start'].replace('-', '')}_{w['end'].replace('-', '')}.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(bundle, separators=(",", ":"), allow_nan=False))
print(f"{len(bundle['runs'])} run(s) [{', '.join(r['policy'] for r in bundle['runs'])}], "
      f"{len(bundle['dates'])} days, {len(bundle['network']['stops']):,} stops -> {out} "
      f"({out.stat().st_size / 1e6:.1f} MB)")
