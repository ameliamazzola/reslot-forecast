"""Compare two sets of sim runs policy by policy; report whether the ranking changed.

Run from the repo folder:

  # before / after (e.g. the lookahead fix): runs from the old commit vs the new one
  python scripts/compare_runs.py --a runs --b data/processed/sim --labels before after

  # sensitivity: the same runs, move_cases inventory_snapshot vs pallet_fill
  python scripts/compare_runs.py --a data/processed/sim --b data/processed/sim \\
         --b-move-cases pallet_fill --labels snapshot pallet_fill

  [--config configs/sim.yaml]   the window compared (default: the config's window)
  [--out DIR]                   default: <paths.out>/compare_<label a>_vs_<label b>_<start>_<end>/

Reads only run_sim.py manifests (src/reslot/sim/compare.py). Primary ranking:
the headline, net labor savings = operator-hours saved vs static x W - move cost
(W = config headline.labor_rate_usd_per_hour), with payback days; secondary: the
notebook's feet scoring. Writes:
  comparison.csv   policy x metric: a, b, change
  manifest.json    git SHA, the manifests compared (hashed), W and its source, the rankings
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from reslot.config import load_config
from reslot.sim import compare as C
from reslot.sim import headline as H
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
    rate, rate_source = H.labor_rate(cfg)
    a, pa = C.load_set(args.a, w["start"], w["end"], args.a_move_cases, rate=rate)
    b, pb = C.load_set(args.b, w["start"], w["end"], args.b_move_cases, rate=rate)
    table, rank = C.compare(a, b, la, lb)
except ValueError as exc:
    raise SystemExit(f"not compared: {exc}")

tag = f"{w['start'].replace('-', '')}_{w['end'].replace('-', '')}"
out = Path(args.out or Path(cfg["paths"]["out"]) / f"compare_{la}_vs_{lb}_{tag}")
out.mkdir(parents=True, exist_ok=True)
table.to_csv(out / "comparison.csv", index=False)
run_cfg = {"seed": cfg.get("seed"), "window": dict(w), "headline": dict(cfg["headline"]),
           **{k: v for k, v in vars(args).items() if k != "out"}}
manifest = write_manifest(out, run_cfg, inputs=[*pa, *pb],
                          extra={"rankings": rank, "primary_ranking": next(iter(rank), None),
                                 "headline": {"definition": H.DEFINITION, "caveat": H.CAVEAT,
                                              "labor_rate_usd_per_hour": rate, "labor_rate_source": rate_source},
                                 "sets": {la: a.to_dict("records"), lb: b.to_dict("records")}})

with pd.option_context("display.width", 200, "display.max_columns", 20):
    print(f"{la}: {args.a} ({args.a_move_cases})   {lb}: {args.b} ({args.b_move_cases})   window {w['start']}..{w['end']}\n")
    print(C.wide(a, b, la, lb).round(1).to_string())
for i, (by, r) in enumerate(rank.items()):
    print(f"\n{'PRIMARY' if i == 0 else 'secondary'} ranking by {by}:  {la}: {' > '.join(r[la])}")
    print(f"{' ' * (len(by) + (23 if i == 0 else 25))}{lb}: {' > '.join(r[lb])}   -> {'CHANGED' if r['changed'] else 'unchanged'}")
dirty = sorted({r['git_sha'] for r in [*a.to_dict('records'), *b.to_dict('records')] if r["git_dirty"]})
if dirty:
    print(f"\nNOTE runs from a dirty tree (not citable): commits {', '.join(dirty)}")
print(f"headline: {H.DEFINITION}; W = ${rate:.2f}/h ({rate_source})")
print(f"caveat: {H.CAVEAT}")
print(f"wrote {out}/")
