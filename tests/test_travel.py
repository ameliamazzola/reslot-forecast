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
