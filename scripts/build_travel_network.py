"""Build the travel network, stop distance table and location crosswalk
from the York DC layout drawing.

Run:  python scripts/build_travel_network.py [configs/travel.yaml]

Reads (data/raw/, gitignored):
  layout_path      DXF with Path_* / Stop_* layers
  locations_path   WMS location master
  picks_path       WMS pick report (coverage check only)

Writes:
  data/processed/travel/stops.csv                    stop_id, x_in, y_in, node_type
  data/processed/travel/stop_distance_ft.csv         stop x stop shortest path, feet
  data/processed/travel/location_stop_crosswalk.csv  location -> stop_id (+ status)
  data/processed/travel/pl_lane_assignment.csv       PL01-PL57 -> lane stops (provisional)
  data/processed/travel/lane_labels.csv              each lane stop -> PL codes it serves
  data/processed/travel/line_stops.csv               production line / supply label -> stop(s)
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
from reslot.travel import lanes as LN
from reslot.travel import network as N
from reslot.utils.manifest import write_manifest

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/travel.yaml")
d, net, sc = cfg["data"], cfg["network"], cfg["scope"]

# --- network ---------------------------------------------------------------
print(f"reading {d['layout_path']} (a few minutes)...")
layout = N.read_layout(d["layout_path"], net["layers"])
G, stops = N.build_graph(layout["lines"], layout["circles"], net["snap_tol_in"])
orphans = N.prune_orphan_fragments(G)
stops, orphan_labels = N.label_stops(stops, layout["labels"], net["label_max_in"])
checks = N.network_checks(G, stops, orphans, layout["misfiled"])
if not checks["connected"] or checks["circles_off_path"] or checks["duplicate_stop_ids"]:
    raise SystemExit(f"network failed checks, fix the drawing first: {checks}")
dist = N.distance_matrix_ft(G, stops)
print(f"{len(layout['lines'])} path lines, {len(stops)} stops {checks['stops_by_type']}, "
      f"connected, dead ends: {checks['dead_ends']}")
if checks["junctions_without_circle"]:
    print(f"  NOTE junctions without a circle: {checks['junctions_without_circle']}")
for k, label in [("duplicate_circles_merged", "duplicate circles merged"),
                 ("orphan_fragments_removed", "stop-less fragments removed"),
                 ("misfiled_entities_ignored", "misfiled entities IGNORED (fix layer in drawing)"),
                 ("rack_stops_at_junctions", "rack circles where paths meet")]:
    v = checks[k]
    if v:
        print(f"  NOTE {label}: {v if isinstance(v, int) else len(v)}"
              + ("" if isinstance(v, int) else f"  e.g. {v[:3]}"))
if orphan_labels:
    print(f"  NOTE labels with no stop within {net['label_max_in']} in (ignored): {orphan_labels}")
if checks["n_dispatch"] != 1:
    raise SystemExit(f"need exactly one Stop_Dispatch circle (trip start), found {checks['n_dispatch']}")
if checks["has_pd"]:
    print("  NOTE a PD_Placeholder circle is still drawn -- retired, delete it")
if net["drop_off_mode"] == "provisional":
    print("  NOTE drop-off lanes follow the agreed PL->lane convention, not a site-confirmed "
          "door map -- label trip totals provisional")

# --- crosswalk -------------------------------------------------------------
slots = load_locations(d["locations_path"])
xw = X.build_crosswalk(slots, stops, cfg["crosswalk"])
load = X.floor_load(xw)
cols = X.column_check(xw, layout["columns"], cfg["crosswalk"])
events = load_wms_picks(d["picks_path"], d["ts_format"])
cov = X.pick_coverage(events, xw, sc["zone"], sc["uom"])
cov_pl = X.pick_coverage(events, xw, "WZGNA", "PL")   # reserve pallet pulls
print(f"crosswalk: {xw['status'].value_counts().to_dict()}")
print(f"  max floor locations per stop {load['max_floor_locations_per_stop']}, "
      f"column sections explained {cols['column_sections_explained']}")
print(f"  {sc['uom']} pick coverage {cov['coverage']:.1%} "
      f"({cov['mapped']:,}/{cov['picks']:,}); unmapped by aisle {cov['top_unmapped_aisles']}")
print(f"  PL pick coverage {cov_pl['coverage']:.1%} of all, {cov_pl['coverage_rack_coded']:.1%} of "
      f"rack-coded (before line stops)")
st_cfg = cfg["staging"]
lanes = LN.lane_stops(stops, st_cfg["group_gap_in"])
pl = LN.pl_lane_counts(slots, st_cfg["pl_pattern"], st_cfg["three_lane_min_capacity"])
assign = LN.assign_pl_lanes(lanes, pl)
labels = LN.lane_labels(lanes, assign)
xw = LN.apply_to_crosswalk(xw, assign)
shared = int((labels["pl_codes"].str.count("/") > 0).sum())
split = assign.loc[assign["split_across_gap"], "location"].tolist()
print(f"  staging: {len(pl)} PL codes asking {int(pl['n_lanes'].sum())} lanes on "
      f"{len(lanes)} lane stops in {lanes['lane_group'].max()} groups; {shared} lanes shared by two PLs; "
      f"split across a group gap: {split}")
dispatch_id = stops.loc[stops["node_type"] == "dispatch", "stop_id"].iloc[0]
xw, line_table = X.apply_labelled_stops(xw, stops, "line", rank_from=dist.loc[dispatch_id])
print("  production lines / supply points: " + ", ".join(
    f"{r.label} ({r.n_stops} stop{'s' if r.n_stops > 1 else ''}{'' if r.in_wms else ', not a WMS code'})"
    for r in line_table.itertuples()))
cov_pl = X.pick_coverage(events, xw, "WZGNA", "PL")   # recount with line pickups mapped
print(f"  PL pick coverage after line stops {cov_pl['coverage']:.1%} of all pallet picks")
by_aisle = xw[xw["aisle"].notna()].groupby("aisle")["status"].value_counts().unstack(fill_value=0)
bad = by_aisle[by_aisle.drop(columns=["ok"], errors="ignore").sum(axis=1) > 0]
if len(bad):
    print("  aisles with unmapped or check_snap locations:\n" + bad.to_string())

# --- write -----------------------------------------------------------------
out = Path(cfg["paths"]["processed"])
out.mkdir(parents=True, exist_ok=True)
stops.drop(columns=["node", "on_path"]).to_csv(out / "stops.csv", index=False)
dist.to_csv(out / "stop_distance_ft.csv")
xw.to_csv(out / "location_stop_crosswalk.csv", index=False)
assign.assign(lane_stop_ids=assign["lane_stop_ids"].str.join("|"),
              lane_groups=assign["lane_groups"].map(lambda g: "|".join(map(str, g)))
              ).to_csv(out / "pl_lane_assignment.csv", index=False)
labels.to_csv(out / "lane_labels.csv", index=False)
line_table.assign(stop_ids=line_table["stop_ids"].str.join("|")).to_csv(out / "line_stops.csv", index=False)

run_dir = Path(cfg["paths"]["runs"]) / "travel_network"
write_manifest(run_dir, cfg, inputs=[d["layout_path"], d["locations_path"], d["picks_path"]],
               extra={"drop_off_mode": net["drop_off_mode"], "network": checks,
                      "staging": {"pl_codes": len(pl), "lanes_asked": int(pl["n_lanes"].sum()),
                                  "lane_stops": len(lanes), "lanes_shared": shared,
                                  "split_across_gap": split},
                      "line_stops": line_table.drop(columns="stop_ids").to_dict("records"),
                      "crosswalk": {"status": xw["status"].value_counts().to_dict(),
                                    **load, **cols},
                      "pick_coverage": cov, "pick_coverage_pl": cov_pl})
print(f"\nwrote {out}/ and {run_dir}/manifest.json")
