"""Generated orders vs real orders over the run window, by total variation distance.

Run:  python scripts/compare_generator.py [configs/sim.yaml] [--mode default|empirical|resample]
                                          [--hour-profile split|single]

Generated: GeneratorSource (config generator, profile generator.profile_path) over
`window` (Oct 12 - Nov 8), drawn without simulating. Real: data.picks_path over
the same window, in-scope case picks, with the profile's excluded SKUs (VMS,
vitamin brands) dropped as they were from the profile. Same definitions on both sides
(src/reslot/sim/generator.py). Statistics, each scored by TVD (0 = identical,
1 = disjoint):

  orders_per_hour        histogram of orders released per clock hour (bins: generator.compare.bins)
  orders_per_hour_weekday / _weekend   the same, Mon-Fri hours and Sat-Sun hours scored separately
  lines_per_order        histogram (tail capped at bins.lines_cap)
  cases_per_line         histogram (tail capped at bins.cases_cap)
  share_1_2_line_orders  |difference of the shares| (TVD of a two-bin distribution)
  sku_concentration      share of lines by SKU rank (SKU identity dropped)

Pass/fail only where generator.compare.thresholds sets a value (set before running).

Noise floor (generator.compare.noise_floor): the generator against itself, seed
pairs (seed_base+1, +2), (+3, +4), ... over the same window and config, the same
statistics; mean and max TVD are reported next to each threshold. A threshold at
or below the noise floor can't be met by any generator; a score above it is not
explained by sampling noise alone.

Writes <paths.out>/generator_compare_<mode>[_<hour profile>]_<start>_<end>/comparison.json + manifest.json.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reslot.config import load_config
from reslot.io.wms_picks import load_wms_picks
from reslot.sim import generator as G
from reslot.sim.engine import window_bounds
from reslot.travel.artifacts import load_travel
from reslot.utils.manifest import write_manifest

ap = argparse.ArgumentParser(description="Generated vs real orders over the run window.")
ap.add_argument("config", nargs="?", default="configs/sim.yaml")
ap.add_argument("--mode", choices=list(G.MODES), help="override generator.mode")
ap.add_argument("--hour-profile", choices=["split", "single"],
                help="override generator.hour_profile (empirical, resample): single = the first version, "
                     "for the before/after")
args = ap.parse_args()
cfg = load_config(args.config)
gcfg = dict(cfg["generator"])
if args.mode:
    gcfg["mode"] = args.mode
if args.hour_profile:
    gcfg["hour_profile"] = args.hour_profile
start, end = window_bounds(cfg)
days = (end - start).days
profile_path = Path(gcfg["profile_path"])
if not profile_path.exists():
    raise SystemExit(f"{profile_path} not found: run scripts/fit_generator_profile.py first")
profile = json.loads(profile_path.read_text(encoding="utf-8"))
net = load_travel(cfg["travel"]["dir"])

# The statistics are order-level (counts, lines, cases, SKUs, release times): visit
# order doesn't enter, so the comparison draws without a routing step.
gcfg["routing"] = {"rule": "draw"}
src = G.make_generator_source(gcfg, cfg["scope"], net, profile, start, days)
n_tours = sum(1 for _ in src.orders())
gen = G.order_stats(G.generator_lines(src), start, end)
picks_path = cfg["data"]["picks_path"]
real_lines = G.order_lines(load_wms_picks(picks_path, cfg["data"]["ts_format"], window=(start, end)),
                           cfg["scope"], start, end)
excl = set(profile.get("excluded_skus", []))
real_kept = G.drop_skus(real_lines, excl)
real = G.order_stats(real_kept, start, end)
scores = G.compare_stats(gen, real, gcfg.get("compare") or {})
nf_cfg = (gcfg.get("compare") or {}).get("noise_floor") or {}
floor = (G.noise_floor(gcfg, cfg["scope"], net, profile, start, end, nf_cfg["seed_pairs"], nf_cfg["seed_base"])
         if nf_cfg.get("seed_pairs") else None)
if floor:
    for k, v in scores.items():
        v["noise_floor_mean"] = floor["scores"][k]["mean"]
        v["noise_floor_max"] = floor["scores"][k]["max"]

w = cfg["window"]
variant = gcfg["mode"] + (f"_{gcfg.get('hour_profile', 'split')}" if gcfg["mode"] != "default" else "")
out = Path(cfg["paths"]["out"]) / f"generator_compare_{variant}_{w['start'].replace('-', '')}_{w['end'].replace('-', '')}"
out.mkdir(parents=True, exist_ok=True)
drop = ("sku_lines",)
res = {"scores": scores, "noise_floor": floor, "generated": {k: v for k, v in gen.items() if k not in drop},
       "real": {k: v for k, v in real.items() if k not in drop}, "source": src.describe(), "definitions": G.DEFINITIONS}
(out / "comparison.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
manifest = write_manifest(out, {**cfg, "generator": gcfg}, inputs=[picks_path, profile_path],
                          extra={"mode": "generator_compare", "generator_mode": gcfg["mode"],
                                 "hour_profile": src.layout.get("hour_profile"), "noise_floor": floor,
                                 "generator_mode_label": src.label, "scores": scores,
                                 "orders_generated": n_tours, "orders_real": real["orders"], "layout": src.layout,
                                 "real_lines_dropped_as_excluded_skus": int(len(real_lines) - len(real_kept))})

print(f"generator: {src.label}; sku_to_face {gcfg.get('sku_to_face')}, {w['start']} .. {w['end']}: "
      f"{gen['orders']:,} generated vs {real['orders']:,} real orders "
      f"({gen['orders_per_day']:.1f} vs {real['orders_per_day']:.1f} per day)")
nf = f"noise floor ({len(floor['seed_pairs'])} seed pairs) mean / max" if floor else "noise floor: not run"
print(f"{'statistic':24s} {'TVD':>7s} {'threshold':>10s}  {'result':6s}  {nf}")
for k, v in scores.items():
    t = "not set" if v["threshold"] is None else f"{v['threshold']:.3f}"
    r = "-" if v["pass"] is None else ("pass" if v["pass"] else "FAIL")
    f = f"{v['noise_floor_mean']:.3f} / {v['noise_floor_max']:.3f}" if floor else ""
    print(f"{k:24s} {v['tvd']:7.3f} {t:>10s}  {r:6s}  {f}")
print(f"share of 1-2 line orders: generated {gen['share_1_2_line_orders']:.1%}, real {real['share_1_2_line_orders']:.1%}")
if manifest["git_dirty"]:
    print("NOTE working tree is dirty: this run is not citable")
print(f"wrote {out}/")
