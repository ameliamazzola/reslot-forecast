"""Tests for the WMS pick/location loaders and the travel calibration.

Several of these guard design decisions, not just correctness. If someone
"simplifies" the trip split to a row split, test_split_no_trip_leak fails and
its docstring says why that matters.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from reslot.io.locations import load_locations, reconcile, slot_inventory
from reslot.io.wms_picks import load_wms_picks
from reslot.schemas import EVENTS, SLOTS, validate
from reslot.travel import calibrate as T
from reslot.travel import crosswalk as X
from reslot.travel import network as N

CFG = {
    "trip": {"group_keys": ["order_id", "user_id"], "min_picks": 3},
    "legs": {"min_seconds": 1, "max_seconds": 900, "max_delta_path": 14000},
}


# --------------------------------------------------------------------------
# Fixtures: a tiny raw extract in C&D's own column names
# --------------------------------------------------------------------------

@pytest.fixture
def raw_picks_csv(tmp_path):
    p = tmp_path / "picks.csv"
    pd.DataFrame({
        "NO": [1, 2, 3],
        "TYPE": ["009", "009", "009"],
        "CREATE_DT": ["08/16/2026 04:37:07 AM", "08/16/2026 04:38:10 AM",
                      "08/31/2026 01:05:00 PM"],
        "USER_NAME": ["U1", "U1", "U2"],
        "FROM_ITEM": ["20000100", "20000100", "00123"],
        "FROM_DESCR": ["a", "a", "b"],
        "FROM_LOCATION": ["FF002A", "FF004A", "JJ119E"],
        "TO_LOCATION": ["PL-01", "PL-01", "RT-19"],
        "FROM_QTY": [3, 5, 1],
        "FROM_UOM": ["CS", "CS", "PL"],
        "TO_QTY": [3, 5, 1],
        "TO_UOM": ["CS", "CS", "PL"],
        "CASES_PER_PALLET": [60, 60, 238],
        "ORD_NO": ["8006855902", "8006855902", "10906764CR"],
        "CUST_NAME": ["CHD VICTORVILLE"] * 3,
        "FROM_FIFO_DT": ["07/29/2026 12:00:00 AM"] * 3,
        "FROM_ZONE": ["WZPCK", "WZPCK", "WZGNJ"],
        "TO_ZONE": ["WZEQP"] * 3,
    }).to_csv(p, index=False)
    return p


@pytest.fixture
def raw_locations_csv(tmp_path):
    p = tmp_path / "locations.csv"
    pd.DataFrame({
        "LOCATION": ["FF002A", "FF004A", "GG010A", "JJ119E"],
        "TYPE": ["F", "F", "F", "R"],
        "WORK_ZONE": ["WZPCK", "WZPCK", "WZPCK", "WZGNJ"],
        "WORK_PATH": [100, 102, 400, 9000],
        "PTWY_ZONE": ["WZPCK"] * 3 + ["WZGNJ"],
        "PTWY_PATH": [1, 2, 3, 4],
        "STD_CPCT": [1, 1, 1, 1],
        "PICK_UOM": ["CASES"] * 3 + ["PALLETS"],
        "X": [None] * 4,
    }).to_csv(p, index=False)
    return p


def make_events_and_slots(n_trips=40, picks_per_trip=6, seed=0):
    """Synthetic stream with a KNOWN travel relationship:
    20s fixed + 5s/case + 40s per aisle change + 0.01s per path unit."""
    rng = np.random.default_rng(seed)
    ev, slot_rows = [], {}
    eid = 0
    for t in range(n_trips):
        clock = pd.Timestamp("2026-09-01") + pd.Timedelta(hours=t)
        path, aisle = 1000.0, "AA"
        for _ in range(picks_per_trip):
            new_path = path + float(rng.integers(1, 400))
            new_aisle = "BB" if rng.random() < 0.2 else aisle
            qty = float(rng.integers(1, 20))
            secs = (20 + 5 * qty + 40 * (new_aisle != aisle)
                    + 0.01 * (new_path - path) + rng.normal(0, 3))
            clock = clock + pd.Timedelta(seconds=max(secs, 1))
            loc = f"{new_aisle}{int(new_path)}A"
            slot_rows[loc] = (new_path, new_aisle)
            ev.append({"event_id": eid, "ts": clock, "event_class": "pick",
                       "from_loc": loc, "qty": qty, "uom": "CS",
                       "order_id": f"O{t}", "user_id": "U1", "from_zone": "WZPCK"})
            eid += 1
            path, aisle = new_path, new_aisle
    events = validate(pd.DataFrame(ev), EVENTS, "events", strict=False)
    slots = pd.DataFrame([{"location": k, "path": v[0], "aisle": v[1], "zone": "WZPCK",
                           "slot_type": "F"} for k, v in slot_rows.items()])
    slots = validate(slots, SLOTS, "slots", strict=False)
    return events, slots


def legs_from(n_trips=40, picks_per_trip=6, seed=0):
    ev, sl = make_events_and_slots(n_trips, picks_per_trip, seed)
    return T.extract_legs(T.pick_face_stream(ev, sl, "WZPCK", "CS"), CFG)


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------

def test_wms_picks_matches_events_contract(raw_picks_csv):
    ev = load_wms_picks(raw_picks_csv)
    validate(ev, EVENTS, "events", strict=False)
    assert (ev["event_class"] == "pick").all()


def test_wms_picks_keeps_string_ids(raw_picks_csv):
    """ORD_NO has alpha suffixes and item codes can have leading zeros.
    Numeric inference would mangle both."""
    ev = load_wms_picks(raw_picks_csv)
    assert "10906764CR" in set(ev["order_id"])
    assert "00123" in set(ev["item_id"])


def test_wms_picks_rejects_wrong_ts_format(raw_picks_csv):
    with pytest.raises(ValueError, match="did not parse"):
        load_wms_picks(raw_picks_csv, ts_format="%Y-%m-%d %H:%M:%S")


def test_locations_contract_and_aisle(raw_locations_csv):
    sl = load_locations(raw_locations_csv)
    validate(sl, SLOTS, "slots", strict=True)
    assert sl.set_index("location").loc["FF002A", "aisle"] == "FF"


def test_reconcile_perfect_and_orphans(raw_picks_csv, raw_locations_csv):
    ev, sl = load_wms_picks(raw_picks_csv), load_locations(raw_locations_csv)
    assert reconcile(ev, sl)["match_rate"] == 1.0
    r = reconcile(ev, sl[sl["location"] != "JJ119E"])
    assert r["match_rate"] < 1.0 and "JJ119E" in r["missing_examples"]


def test_slot_inventory_keeps_untouched_faces(raw_picks_csv, raw_locations_csv):
    """GG010A never appears in a transaction. It is free capacity, and losing
    it would delete the re-slotting headroom the project depends on."""
    sl = load_locations(raw_locations_csv)
    inv = slot_inventory(sl, "WZPCK", "F")
    assert "GG010A" in set(inv["location"]) and len(inv) == 3


# --------------------------------------------------------------------------
# Legs and split
# --------------------------------------------------------------------------

def test_min_picks_respected():
    assert len(legs_from(n_trips=5, picks_per_trip=2)) == 0


def test_one_leg_fewer_than_picks_per_trip():
    assert len(legs_from(n_trips=10, picks_per_trip=5)) == 10 * 4


def test_split_no_trip_leak():
    """Legs in one trip share an operator, an hour and an aisle. A leg-level
    split leaks and returns a flattering holdout error."""
    legs = legs_from(n_trips=40)
    tr, te = T.split_by_trip(legs, 0.25, seed=1)
    assert not set(tr["trip_id"]) & set(te["trip_id"])
    assert len(tr) + len(te) == len(legs)


def test_split_is_seeded():
    legs = legs_from(n_trips=40)
    pd.testing.assert_frame_equal(T.split_by_trip(legs, 0.25, 7)[0],
                                  T.split_by_trip(legs, 0.25, 7)[0])


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def test_recovers_known_coefficients():
    """Evidence the method works, not just that it runs."""
    m = T.TravelModel.fit(legs_from(300, 8), n_bins=10, isotonic_residual=False)
    assert m.coef["per_case_s"] == pytest.approx(5.0, abs=0.6)
    assert m.coef["cross_aisle_s"] == pytest.approx(40.0, abs=8.0)
    assert m.coef["fixed_s"] == pytest.approx(20.0, abs=6.0)


def test_beats_constant_and_distance_only():
    legs = legs_from(300, 8)
    tr, te = T.split_by_trip(legs, 0.25, seed=3)
    r = T.evaluate(T.TravelModel.fit(tr, 10), te, float(tr["elapsed_s"].median()),
                   T.fit_distance_only(tr, 10))
    assert r["full_model"]["mae"] < r["distance_only"]["mae"]
    assert r["full_model"]["mae"] < r["constant"]["mae"]


def test_distance_term_monotone():
    """Further is never faster -- the only physical claim the fit encodes."""
    m = T.TravelModel.fit(legs_from(200, 8), n_bins=12)
    assert np.all(np.diff(m._travel(np.arange(0, 2000, 100))) >= -1e-9)


def test_move_cost_excludes_case_handling():
    """A re-slot pays travel, not per-case pick time."""
    m = T.TravelModel.fit(legs_from(200), n_bins=10)
    leg = pd.DataFrame({"qty": [10.0], "delta_path": [500.0], "cross_aisle": [0]})
    assert m.move_cost(500.0, False) < m.predict(leg)[0]
    assert m.move_cost(500.0, True) > m.move_cost(500.0, False)


def test_results_carry_proxy_source():
    """Same convention as panel source='pick_proxy': the caveat has to travel
    with the number into the manifest and the report."""
    legs = legs_from(100)
    tr, te = T.split_by_trip(legs, 0.25, seed=5)
    m = T.TravelModel.fit(tr, 10)
    for r in (T.evaluate(m, te, float(tr["elapsed_s"].median())),
              T.decompose(m, te), T.trip_check(m, te)):
        assert r["source"] == T.SOURCE and r["caveat"]
    assert (T.cost_table(m, 1000.0)["source"] == T.SOURCE).all()


def test_decomposition_sums_to_one():
    m = T.TravelModel.fit(legs_from(200), n_bins=10)
    d = T.decompose(m, legs_from(200))
    parts = ["fixed_handling", "case_handling", "path_travel", "cross_aisle", "unexplained"]
    assert sum(d[k] for k in parts) == pytest.approx(1.0)


def test_pick_proxy_warns_when_no_eaches(raw_picks_csv):
    """The pick report has no eaches. Silently zero qty made every SKU look
    like it was never picked (zero-day share 1.00)."""
    from reslot.features import panel as P
    ev = load_wms_picks(raw_picks_csv)
    with pytest.warns(UserWarning, match="qty_eaches is null"):
        p = P.from_pick_proxy(ev)
    assert (P.intermittency(p, value_col="n_lines")["zero_share"] < 1.0).all()


# --------------------------------------------------------------------------
# Network (synthetic layout -- no drawing needed)
# --------------------------------------------------------------------------
# Two aisles 236" apart, cross-aisles top and bottom, one rack stop mid-aisle
# on each. Same geometry style as the York drawing, small enough to check by hand.

def tiny_layout():
    lines = [((0, 0), (0, 1000), "path_aisle"), ((236, 0), (236, 1000), "path_aisle"),
             ((0, 1000), (236, 1000), "path_cross"), ((0, 0), (236, 0), "path_cross")]
    circles = [(0, 1000, "junction"), (236, 1000, "junction"), (0, 0, "junction"),
               (236, 0, "junction"), (0, 500, "rack"), (236, 500, "rack")]
    return lines, circles


def test_network_distances_by_hand():
    G, stops = N.build_graph(*tiny_layout())
    D = N.distance_matrix_ft(G, stops)
    # rack to rack: up 500, across 236, down 500 (either way round)
    assert D.loc["S_0.0_500.0", "S_236.0_500.0"] == pytest.approx((500 + 236 + 500) / 12, abs=0.01)
    assert (D.values == D.values.T).all() and (D.values.diagonal() == 0).all()


def test_circle_mid_line_becomes_node():
    """Stops are drawn as circles ON a continuous aisle line; the builder must
    split the line there or the stop is unreachable."""
    G, stops = N.build_graph(*tiny_layout())
    assert stops["on_path"].all()
    assert N.network_checks(G, stops)["connected"]


def test_crossing_without_shared_point_is_not_a_junction():
    """Lines that merely cross are NOT joined -- a truck can only turn where the
    drawing says so. If someone adds auto-intersection, this fails on purpose:
    it would silently connect things that are not connected (e.g. across a
    rack end with no opening)."""
    lines = [((0, 0), (0, 100), "path_aisle"), ((-50, 50), (50, 50), "path_cross")]
    G, _ = N.build_graph(lines, [])
    import networkx as nx
    assert nx.number_connected_components(G) == 2


def test_checks_flag_junction_without_circle():
    lines, circles = tiny_layout()
    lines = lines + [((118, 1000), (118, 1300), "path_aisle")]   # spur off the top, no circle
    G, stops = N.build_graph(lines, circles + [(118, 1300, "lane")])
    c = N.network_checks(G, stops)
    assert c["junctions_without_circle"] == ["118.0,1000.0"]
    # a lane mouth at the end of its spur is a dead end by design, not a finding
    assert c["dead_ends"] == []

    G, stops = N.build_graph(lines, circles)                       # same spur, no stop at its end
    assert "vertex@118.0,1300.0" in N.network_checks(G, stops)["dead_ends"]


def test_duplicate_circles_are_one_stop():
    """A circle pasted twice at the same point is one stop, and the count of
    merged copies is reported (the full-building drawing has 108)."""
    lines, circles = tiny_layout()
    G, stops = N.build_graph(lines, circles + [(0, 500, "rack"), (0, 500, "rack")])
    assert stops["stop_id"].is_unique
    assert N.network_checks(G, stops)["duplicate_circles_merged"] == 2


def test_stopless_fragment_is_pruned_but_stopped_piece_is_not():
    """A stray 1-inch line can't change any route, so it is dropped and listed.
    A disconnected piece that carries a stop is a real gap: it stays, and the
    connectivity check fails."""
    lines, circles = tiny_layout()
    stray = [((500, 500), (500, 501), "path_aisle")]
    G, stops = N.build_graph(lines + stray, circles)
    removed = N.prune_orphan_fragments(G)
    assert len(removed) == 1 and N.network_checks(G, stops, removed)["connected"]

    island = [((900, 0), (900, 100), "path_aisle")]
    G, stops = N.build_graph(lines + island, circles + [(900, 50, "rack")])
    assert N.prune_orphan_fragments(G) == []
    assert not N.network_checks(G, stops)["connected"]


# --------------------------------------------------------------------------
# Labels and production-line stops
# --------------------------------------------------------------------------

def _line_stops():
    return pd.DataFrame({"stop_id": ["LN_0.0_0.0", "LN_0.0_100.0", "LN_500.0_0.0", "J_0.0_50.0"],
                         "x_in": [0.0, 0.0, 500.0, 0.0], "y_in": [0.0, 100.0, 0.0, 50.0],
                         "node_type": ["line", "line", "line", "junction"]})


def test_label_goes_to_nearest_stop_within_reach_and_never_a_junction():
    labels = [(2, 3, "L6", "line_label"), (1, 98, "L6", "line_label"),
              (0, 52, "X", "line_label"),          # sits on the junction: nearest non-junction is 48 in away
              (900, 900, "L9", "line_label")]      # nowhere near a stop
    st, orphans = N.label_stops(_line_stops(), labels, max_in=24)
    assert st.set_index("stop_id")["label"].to_dict() == {
        "LN_0.0_0.0": "L6", "LN_0.0_100.0": "L6", "LN_500.0_0.0": None, "J_0.0_50.0": None}
    assert orphans == ["X@0.0,52.0", "L9@900.0,900.0"]


def test_code_drawn_twice_maps_to_the_nearer_pickup():
    st = _line_stops().assign(label=["L6", "L6", "Slip sheets", None])
    xw = pd.DataFrame({"location": ["L6", "C125B"], "status": ["not_a_rack_code", "ok"]})
    rank = pd.Series({"LN_0.0_0.0": 300.0, "LN_0.0_100.0": 120.0, "LN_500.0_0.0": 50.0})
    xw, table = X.apply_labelled_stops(xw, st, "line", rank_from=rank)
    assert xw.set_index("location").loc["L6", "stop_id"] == "LN_0.0_100.0"
    assert xw.set_index("location").loc["L6", "status"] == "ok"
    t = table.set_index("label")
    assert t.loc["L6", "n_stops"] == 2 and t.loc["L6", "in_wms"]
    assert not t.loc["Slip sheets", "in_wms"]


# --------------------------------------------------------------------------
# Staging lanes (PL codes -> lane stops)
# --------------------------------------------------------------------------

from reslot.travel import lanes as LN


def _lanes(groups):
    """Lane stops: `groups` lane groups of 4, 50 in apart, 200 in between groups,
    numbered from the right like the drawing."""
    xs, x = [], 10_000.0
    for _ in range(groups):
        for _ in range(4):
            xs.append(x); x -= 50
        x -= 150
    st = pd.DataFrame({"stop_id": [f"L_{v:.1f}_0.0" for v in xs], "x_in": xs,
                       "y_in": 0.0, "node_type": "lane"})
    return LN.lane_stops(st, group_gap_in=60)


def _pl(caps):
    return pd.DataFrame({"location": [f"PL{i:02d}" for i in range(1, len(caps) + 1)],
                         "capacity": caps,
                         "n_lanes": [3 if c >= 66 else 2 for c in caps]})


def test_lane_numbering_runs_right_to_left_by_group():
    ln = _lanes(2)
    assert ln["lane_no"].tolist() == list(range(1, 9))
    assert ln["x_in"].is_monotonic_decreasing
    assert ln["lane_group"].tolist() == [1, 1, 1, 1, 2, 2, 2, 2]


def test_pl_assignment_covers_every_lane_in_order_and_shares_at_most_one():
    """Three PLs asking 2+2+3 = 7 lanes on 6 lanes: one shared lane, PL01 on
    lane 1, the last PL on the last lane, order preserved."""
    ln = _lanes(2)[lambda d: d["lane_no"] <= 6]
    a = LN.assign_pl_lanes(ln, _pl([44, 44, 66]))
    assert a["first_lane"].iloc[0] == 1 and a["last_lane"].iloc[-1] == 6
    assert a["first_lane"].is_monotonic_increasing
    gaps = a["first_lane"].iloc[1:].to_numpy() - a["last_lane"].iloc[:-1].to_numpy()
    assert set(gaps) <= {0, 1}                      # 0 = shares one lane, 1 = adjacent
    labels = LN.lane_labels(ln, a)
    assert (labels["pl_codes"] != "").all()
    assert labels["pl_codes"].str.count("/").max() <= 1


def test_pl_assignment_keeps_pls_inside_a_group_when_it_can():
    """Two groups of 4, four 2-lane PLs: the clean answer is two PLs per group
    and nobody straddles the gap."""
    a = LN.assign_pl_lanes(_lanes(2), _pl([44, 44, 44, 44]))
    assert not a["split_across_gap"].any()
    assert a[["first_lane", "last_lane"]].values.tolist() == [[1, 2], [3, 4], [5, 6], [7, 8]]


def test_pl_assignment_refuses_when_lanes_cannot_be_covered():
    with pytest.raises(ValueError):
        LN.assign_pl_lanes(_lanes(3), _pl([44, 44]))      # 4 lanes asked, 12 drawn


def test_two_lane_pl_uses_right_lane_three_lane_pl_uses_middle():
    ln = _lanes(2)[lambda d: d["lane_no"] <= 5]
    a = LN.assign_pl_lanes(ln, _pl([44, 66]))
    by = dict(zip(ln["lane_no"], ln["stop_id"]))
    assert a["stop_id"].tolist() == [by[1], by[a["first_lane"].iloc[1] + 1]]


# --------------------------------------------------------------------------
# Crosswalk rules
# --------------------------------------------------------------------------

import yaml as _yaml

XCFG = _yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "travel.yaml")
                       .read_text())["crosswalk"]
R = XCFG["bay_rules"]


def ty(b):
    return X.target_y(b, R["segments"], R["pitch_in"], R["bays_per_stop"])


def test_four_bays_per_stop_from_the_dock():
    """Bays 1-4 share the first stop at the top; 5 starts the next section."""
    assert ty(1)[1] == ty(4)[1] == R["segments"][0]["first_y"]
    assert ty(5)[1] == pytest.approx(ty(1)[1] - R["pitch_in"])


def test_tunnel_bays_map_to_cross_aisle_junction():
    """79-84 and 185-190 exist only above the cross-aisles; the truck works them
    from the junction underneath, and they must not step down the aisle."""
    for b in (79, 84, 185, 190):
        seg, y, node = ty(b)
        assert seg.startswith("tunnel") and node == "junction"
    assert ty(79)[1] == ty(84)[1]


def test_segments_are_anchored_not_counted():
    """Each third restarts at its own anchor, so a counting error in one third
    cannot shift the next."""
    assert ty(85)[1] == [s for s in R["segments"] if s["name"] == "middle"][0]["first_y"]
    assert ty(191)[1] == [s for s in R["segments"] if s["name"] == "bottom"][0]["first_y"]


def test_segments_cover_every_bay_once():
    covered = [b for s in R["segments"] for b in range(s["first_bay"], s["last_bay"] + 1)]
    assert sorted(covered) == list(range(1, R["max_bay"] + 1))


# --------------------------------------------------------------------------
# Real outputs -- skipped unless scripts/build_travel_network.py has been run
# --------------------------------------------------------------------------

PROC = Path(__file__).resolve().parents[1] / "data" / "processed" / "travel"
needs_outputs = pytest.mark.skipif(not (PROC / "stop_distance_ft.csv").exists(),
                                   reason="run scripts/build_travel_network.py first (needs C&D data)")


@needs_outputs
@pytest.mark.parametrize("a,b,feet", [
    ("D_183183.9_5933.3", "S_180855.6_5571.4", 224.2),    # dispatch -> FF bays 1-4: (223+2328+139)/12
    ("D_183183.9_5933.3", "L_188380.6_5766.6", 456.3),    # dispatch -> lane 1 (PL01): (223+5197+56)/12
    ("D_183183.9_5933.3", "LN_185783.2_3290.3", 436.9),   # dispatch -> line L3 via Y and the upper cross-aisle:
                                                          # (223+77+2184+2400+237+123)/12
    ("S_180855.6_926.4", "S_180619.6_926.4", 41.1),       # FF -> GG via middle cross-aisle
    ("S_180855.6_3398.4", "S_180619.6_3398.4", 41.1),     # FF -> GG via upper cross-aisle
])
def test_real_routes_match_hand_calcs(a, b, feet):
    """Hand-traced on the drawing. If one moves, the drawing or the builder changed."""
    D = pd.read_csv(PROC / "stop_distance_ft.csv", index_col=0)
    assert D.loc[a, b] == pytest.approx(feet, abs=0.1)


@needs_outputs
@pytest.mark.parametrize("a,b,feet", [
    ("S_181208.6_926.4", "S_180855.6_926.4", 50.8),     # EE -> FF via middle cross-aisle
    ("S_190784.6_5571.4", "S_180855.6_5571.4", 850.6),  # A top -> FF top, along the dock aisle
])
def test_real_east_routes_match_hand_calcs(a, b, feet):
    """East block, hand-traced: (128.5+353+128.5)/12 and (139+9929+139)/12."""
    D = pd.read_csv(PROC / "stop_distance_ft.csv", index_col=0)
    assert D.loc[a, b] == pytest.approx(feet, abs=0.1)


@needs_outputs
def test_real_pl_assignment_matches_agreed_rules():
    """PL01 on the rightmost lane, PL57 on the leftmost, every lane serves at
    least one and at most two PLs, lane counts follow capacity."""
    a = pd.read_csv(PROC / "pl_lane_assignment.csv")
    lab = pd.read_csv(PROC / "lane_labels.csv")
    assert a["location"].tolist() == [f"PL{i:02d}" for i in range(1, 58)]
    assert a.loc[0, "first_lane"] == 1 and a.iloc[-1]["last_lane"] == len(lab)
    assert (a["n_lanes"] == a["capacity"].map(lambda c: 3 if c >= 66 else 2)).all()
    assert lab["pl_codes"].notna().all() and lab["pl_codes"].str.count("/").max() <= 1


@needs_outputs
def test_real_production_lines_all_have_a_stop():
    lines = pd.read_csv(PROC / "line_stops.csv").set_index("label")
    for code in [f"L{i}" for i in range(1, 8)]:
        assert lines.loc[code, "in_wms"], code
    xw = pd.read_csv(PROC / "location_stop_crosswalk.csv", low_memory=False).set_index("location")
    assert (xw.loc[[f"L{i}" for i in range(1, 8)], "status"] == "ok").all()


@needs_outputs
def test_real_crosswalk_never_overloads_a_stop():
    xw = pd.read_csv(PROC / "location_stop_crosswalk.csv", dtype={"level": str})
    assert X.floor_load(xw)["max_floor_locations_per_stop"] <= R["bays_per_stop"]
