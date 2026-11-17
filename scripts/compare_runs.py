"""Compare two sets of sim runs policy by policy; report whether the ranking changed.

Run from the repo folder:

  # before / after (e.g. the lookahead fix): runs from the old commit vs the new one
  python scripts/compare_runs.py --a runs --b data/processed/sim --labels before after

  # sensitivity: the same runs, move_cases inventory_snapshot vs pallet_fill
  python scripts/compare_runs.py --a data/processed/sim --b data/processed/sim \\
         --b-move-cases pallet_fill --labels snapshot pallet_fill

  [--config configs/sim.yaml]   the window compared (default: the config's window)
  [--out DIR]                   default: <paths.out>/compare_<label a>_vs_<label b>_<start>_<end>/

Reads only run_sim.py manifests (src/reslot/sim/compare.py). Writes:
  comparison.csv   policy x metric: a, b, change
  manifest.json    git SHA, the manifests compared (hashed), both rankings
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from reslot.config import load_config
from reslot.sim import compare as C
from reslot.utils.manifest import write_manifest

ap = argparse.ArgumentParser(description="Compare two sets of sim runs and their policy ranking.")
ap.add_argument("--a", required=True, help="directory holding the first set of sim_* run folders")
ap.add_argument("--b", required=True, help="directory holding the second set")
ap.add_argument("--a-move-cases", default="inventory_snapshot", choices=sorted(C.VARIANT_SUFFIX))
ap.add_argument("--b-move-cases", default="inventory_snapshot", choices=sorted(C.VARIANT_SUFFIX))
ap.add_argument("--labels", nargs=2, default=["a", "b"], metavar=("A", "B"))
ap.add_argument("--config", default="configs/sim.yaml")
ap.add_argument("--out")
args = ap.parse_args()

cfg = load_config(args.config)
w = cfg["window"]
la, lb = args.labels
try:
    a, pa = C.load_set(args.a, w["start"], w["end"], args.a_move_cases)
    b, pb = C.load_set(args.b, w["start"], w["end"], args.b_move_cases)
    table, rank = C.compare(a, b, la, lb)
except ValueError as exc:
    raise SystemExit(f"not compared: {exc}")

tag = f"{w['start'].replace('-', '')}_{w['end'].replace('-', '')}"
out = Path(args.out or Path(cfg["paths"]["out"]) / f"compare_{la}_vs_{lb}_{tag}")
out.mkdir(parents=True, exist_ok=True)
table.to_csv(out / "comparison.csv", index=False)
run_cfg = {"seed": cfg.get("seed"), "window": dict(w), **{k: v for k, v in vars(args).items() if k != "out"}}
manifest = write_manifest(out, run_cfg, inputs=[*pa, *pb],
                          extra={"rankings": rank, "sets": {la: a.to_dict("records"), lb: b.to_dict("records")}})

with pd.option_context("display.width", 200, "display.max_columns", 20):
    print(f"{la}: {args.a} ({args.a_move_cases})   {lb}: {args.b} ({args.b_move_cases})   window {w['start']}..{w['end']}\n")
    print(C.wide(a, b, la, lb).round(1).to_string())
for by, r in rank.items():
    print(f"\nranking by {by}:  {la}: {' > '.join(r[la])}")
    print(f"{' ' * (len(by) + 13)}{lb}: {' > '.join(r[lb])}   -> {'CHANGED' if r['changed'] else 'unchanged'}")
dirty = sorted({r['git_sha'] for r in [*a.to_dict('records'), *b.to_dict('records')] if r["git_dirty"]})
if dirty:
    print(f"\nNOTE runs from a dirty tree (not citable): commits {', '.join(dirty)}")
print("sim_busy is not a finding while dispatch legs are untimed (decisions 2026-11-02)")
print(f"wrote {out}/")
