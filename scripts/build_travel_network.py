"""Build the travel network, stop distance table and location crosswalk
from the York DC layout drawing.

Run:  python scripts/build_travel_network.py [configs/travel.yaml]

Reads (data/raw/, gitignored):
  layout_path      DXF with Path_* / Stop_* / PD_* layers
  locations_path   WMS location master
  picks_path       WMS pick report (coverage check only)

Writes:
  data/processed/travel/stops.csv                    stop_id, x_in, y_in, node_type
  data/processed/travel/stop_distance_ft.csv         stop x stop shortest path, feet
  data/processed/travel/location_stop_crosswalk.csv  location -> stop_id (+ status)
  runs/travel_network/manifest.json                  hashes, checks, coverage

All outputs are derived from C&D data and stay out of git.
Reading the DXF takes a few minutes (the file is ~240 MB).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reslot.config import load_config
from reslot.io.locations import load_locations
from reslot.io.wms_picks import load_wms_picks
from reslot.travel import crosswalk as X
from reslot.travel import network as N
from reslot.utils.manifest import write_manifest

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/travel.yaml")
d, net, sc = cfg["data"], cfg["network"], cfg["scope"]

# --- network ---------------------------------------------------------------
print(f"reading {d['layout_path']} (a few minutes)...")
layout = N.read_layout(d["layout_path"], net["layers"])
G, stops = N.build_graph(layout["lines"], layout["circles"], net["snap_tol_in"])
checks = N.network_checks(G, stops)
if not checks["connected"] or checks["circles_off_path"] or checks["duplicate_stop_ids"]:
    raise SystemExit(f"network failed checks, fix the drawing first: {checks}")
dist = N.distance_matrix_ft(G, stops)
print(f"{len(layout['lines'])} path lines, {len(stops)} stops {checks['stops_by_type']}, "
      f"connected, dead ends: {checks['dead_ends']}")
if checks["junctions_without_circle"]:
    print(f"  NOTE junctions without a circle: {checks['junctions_without_circle']}")
if net["pd_mode"] == "placeholder":
    print("  NOTE trip start/drop-off is a PLACEHOLDER -- do not cite trip totals")

# --- crosswalk -------------------------------------------------------------
slots = load_locations(d["locations_path"])
xw = X.build_crosswalk(slots, stops, cfg["crosswalk"])
load = X.floor_load(xw)
cols = X.column_check(xw, layout["columns"], cfg["crosswalk"])
events = load_wms_picks(d["picks_path"], d["ts_format"])
cov = X.pick_coverage(events, xw, sc["zone"], sc["uom"])
print(f"crosswalk: {xw['status'].value_counts().to_dict()}")
print(f"  max floor locations per stop {load['max_floor_locations_per_stop']}, "
      f"column sections explained {cols['column_sections_explained']}")
print(f"  {sc['uom']} pick coverage {cov['coverage']:.1%} "
      f"({cov['mapped']:,}/{cov['picks']:,}); unmapped by aisle {cov['top_unmapped_aisles']}")

# --- write -----------------------------------------------------------------
out = Path(cfg["paths"]["processed"])
out.mkdir(parents=True, exist_ok=True)
stops.drop(columns=["node", "on_path"]).to_csv(out / "stops.csv", index=False)
dist.to_csv(out / "stop_distance_ft.csv")
xw.to_csv(out / "location_stop_crosswalk.csv", index=False)

run_dir = Path(cfg["paths"]["runs"]) / "travel_network"
write_manifest(run_dir, cfg, inputs=[d["layout_path"], d["locations_path"], d["picks_path"]],
               extra={"pd_mode": net["pd_mode"], "network": checks,
                      "crosswalk": {"status": xw["status"].value_counts().to_dict(),
                                    **load, **cols},
                      "pick_coverage": cov})
print(f"\nwrote {out}/ and {run_dir}/manifest.json")
