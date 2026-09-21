"""Calibrate the travel-time model from the C&D WMS extracts.

Run:  python scripts/calibrate_travel.py [configs/travel.yaml]

Writes:
  data/processed/travel/travel_profile.csv   observed seconds by distance bin
  data/processed/travel/travel_cost.csv      repositioning cost table for the sim
  runs/travel_seed<seed>/manifest.json       config hash, git SHA, input hashes, metrics

Both output folders are gitignored: everything here is derived from C&D data.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reslot.config import load_config
from reslot.io.locations import load_locations, reconcile, slot_inventory
from reslot.io.wms_picks import load_wms_picks
from reslot.travel import calibrate as T
from reslot.utils.manifest import write_manifest

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/travel.yaml")
d, sc = cfg["data"], cfg["scope"]

# --- load + reconcile -----------------------------------------------------
slots = load_locations(d["locations_path"])
events = load_wms_picks(d["picks_path"], d["ts_format"])
recon = reconcile(events, slots)
print(f"reconciliation: {recon['matched']:,}/{recon['transacted_locations']:,} "
      f"transacted locations in master ({recon['match_rate']:.1%})")
if recon["match_rate"] < 1.0:
    print(f"  WARNING unmatched: {recon['missing_examples']}")

stream = T.pick_face_stream(events, slots, sc["zone"], sc["uom"])
faces = slot_inventory(slots, sc["zone"], sc["slot_type"])
touched = int(stream["from_loc"].nunique())
print(f"{len(stream):,} case picks across {touched:,} faces; "
      f"{len(faces):,} faces in building ({len(faces) - touched:,} free)")

# --- legs + split ---------------------------------------------------------
legs = T.extract_legs(stream, cfg)
train, test = T.split_by_trip(legs, cfg["validation"]["holdout_fraction"], cfg["seed"])
print(f"{len(legs):,} legs from {legs['trip_id'].nunique():,} trips "
      f"-> train {len(train):,} / holdout {len(test):,} (split by trip)")

# --- fit + evaluate -------------------------------------------------------
nb = cfg["fit"]["n_bins"]
model = T.TravelModel.fit(train, n_bins=nb, isotonic_residual=cfg["fit"]["isotonic_residual"])
dist_only = T.fit_distance_only(train, nb)
metrics = T.evaluate(model, test, float(train["elapsed_s"].median()), dist_only)
shares = T.decompose(model, test)
trips = T.trip_check(model, test)

# --- write ----------------------------------------------------------------
out = Path(cfg["paths"]["processed"])
out.mkdir(parents=True, exist_ok=True)
T.distance_profile(train, nb).to_csv(out / "travel_profile.csv", index=False)
T.cost_table(model, float(legs["delta_path"].max()),
             cfg["output"]["cost_table_step"]).to_csv(out / "travel_cost.csv", index=False)

run_dir = Path(cfg["paths"]["runs"]) / f"travel_seed{cfg['seed']}"
write_manifest(run_dir, cfg, inputs=[d["picks_path"], d["locations_path"]], extra={
    "reconciliation": recon,
    "scope": {**sc, "picks": len(stream), "faces_touched": touched,
              "faces_in_building": len(faces)},
    "legs": {"n": len(legs), "n_trips": int(legs["trip_id"].nunique()),
             "n_train": len(train), "n_holdout": len(test)},
    "coefficients": model.coef,
    "holdout": metrics,
    "time_decomposition": shares,
    "trip_level": trips,
})

# --- report ---------------------------------------------------------------
print("\ncoefficients (seconds)")
for k, v in model.coef.items():
    print(f"  {k:20s} {v:9.3f}")
print("\nholdout MAE (seconds)")
for k in ("constant", "distance_only", "full_model"):
    print(f"  {k:20s} {metrics[k]['mae']:9.1f}")
print("\nshare of observed pick time")
for k, v in shares.items():
    if isinstance(v, float):
        print(f"  {k:28s} {v:7.1%}")
print(f"\ntrip level: median abs pct error {trips['median_abs_pct_error']:.1%}, "
      f"spearman {trips['spearman']:.3f}")
print(f"\n[{metrics['source']}] {metrics['caveat']}")
print(f"\nwrote {out}/ and {run_dir}/manifest.json")
