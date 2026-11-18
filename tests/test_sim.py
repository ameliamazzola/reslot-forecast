"""Tests for the SimPy replay (travel/artifacts, sim/workload, sim/policy,
sim/engine, scripts/run_sim.py). Synthetic data only -- no C&D files.

Synthetic network (feet):        D    S1   S2   S3
                            D    0    10   25   40
                            S1  10     0   15   30
                            S2  25    15    0   20
                            S3  40    30   20    0
Faces: FF001A -> S1 (aisle FF), FF003A -> S2 (FF), GG001A -> S3 (GG).
Synthetic leg model: 10 + 2*qty + 30*[aisle changed] + 0.1*feet seconds.
None of these numbers are the real model.
"""
import json
import runpy
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest
import yaml

from reslot.schemas import EVENTS, validate
from reslot.sim import engine as E
from reslot.sim.policy import Move, StaticPolicy, make_policy
from reslot.sim.workload import Pick, Tour, build_workload
from reslot.travel.artifacts import (ArtifactError, TravelNetwork, load_crosswalk,
                                     load_distance_table, load_travel)
from reslot.utils.manifest import write_manifest

REPO = Path(__file__).resolve().parents[1]
START, END = pd.Timestamp("2026-10-05"), pd.Timestamp("2026-10-12")
SCOPE = {"event_class": "pick", "zone": "WZPCK", "uom": "CS", "slot_type": "F"}
TOUR = {"group_keys": ["order_id", "user_id"]}
ENDS = {"start_stop": "D", "end_stop": "D", "dispatch_legs": "untimed", "drop_off_mode": "provisional"}
CHECK = {"min_picks": 3, "leg_min_seconds": 1, "leg_max_seconds": 900}
STOPS = ["D", "S1", "S2", "S3"]
FEET = [[0, 10, 25, 40], [10, 0, 15, 30], [25, 15, 0, 20], [40, 30, 20, 0]]


class Synthetic:
    name = "synthetic"

    def __init__(self, cfg=None):
        pass

    def predict(self, leg):
        return 10 + 2 * leg.qty + 30 * leg.cross_aisle + 0.1 * leg.feet


def distances():
    return pd.DataFrame(FEET, dtype=float, index=pd.Index(STOPS, name="stop_id"), columns=STOPS)


def crosswalk():
    return pd.DataFrame({
        "location":  ["FF001A", "FF003A", "GG001A", "XY160A", "PL01", "FF005A", "007", "J100A"],
        "aisle":     ["FF", "FF", "GG", "XY", None, "FF", None, "J"],
        "zone":      ["WZPCK", "WZPCK", "WZPCK", "WZPCK", "WZNDK", "WZPCK", "WZPCK", "WZGNA"],
        "slot_type": ["F", "F", "F", "F", "R", "F", "F", "R"],
        "status":    ["ok", "ok", "ok", "check_snap", "ok_provisional", "ok", "not_a_rack_code", "ok"],
        "stop_id":   ["S1", "S2", "S3", "S2", "L_9", "S_gone", None, "S1"]})


@pytest.fixture
def net():
    return TravelNetwork(distances(), crosswalk())


@pytest.fixture
def travel_dir(tmp_path):
    """Both files written the way scripts/build_travel_network.py writes them."""
    distances().to_csv(tmp_path / "stop_distance_ft.csv")
    crosswalk().to_csv(tmp_path / "location_stop_crosswalk.csv", index=False)
    return tmp_path


def events(rows):
    """rows: (order, user, 'YYYY-MM-DD HH:MM:SS', location, qty[, zone, uom, event_class])"""
    out = []
    for i, r in enumerate(rows, start=1):
        o, u, ts, loc, q, *rest = r
        zone, uom, cls = (rest + ["WZPCK", "CS", "pick"][len(rest):])[:3]
        out.append({"event_id": i, "ts": pd.Timestamp(ts), "event_class": cls, "order_id": o,
                    "user_id": u, "item_id": "SKU1", "from_loc": loc, "qty": q,
                    "from_zone": zone, "uom": uom})
    return validate(pd.DataFrame(out), EVENTS, "events", strict=False)


def tour(*locs, qtys=None, tid=0, arrival=0.0, n_dropped=0):
    ts = START + pd.Timedelta(hours=8)
    qtys = qtys or [1.0] * len(locs)
    return Tour(tid, f"O{tid}", f"U{tid}", arrival,
                tuple(Pick(i + 1, ts, l, float(q), None) for i, (l, q) in enumerate(zip(locs, qtys))),
                n_dropped)


def sim(tours, net, operators=6, policy=None, model=None, ends=ENDS, **kw):
    return E.simulate(tours, operators, policy or make_policy("static"), net, model or Synthetic(),
                      SCOPE, ends, **kw)


def cfg(**over):
    c = {"seed": 1, "window": {"start": "2026-10-05", "end": "2026-10-11"}, "operators": 6,
         "policy": "static", "scope": dict(SCOPE), "tour": dict(TOUR), "leg_time": {"model": None},
         "trip_ends": dict(ENDS), "leg_model_check": dict(CHECK),
         "headline": {"labor_rate_usd_per_hour": 45.5, "labor_rate_source": "test rate"}}
    c.update(over)
    return c


# --------------------------------------------------------------------------
# travel/artifacts
# --------------------------------------------------------------------------

def test_artifacts_read_the_build_script_format(travel_dir):
    n = load_travel(travel_dir)
    assert n.feet("D", "S3") == 40 and n.feet("S2", "S1") == 15
    assert set(n.sources) == {"distance_table", "crosswalk"}
    assert load_crosswalk(travel_dir / "location_stop_crosswalk.csv").loc[6, "location"] == "007"


def test_feet_is_a_lookup_and_unknown_stops_raise(net):
    with pytest.raises(KeyError, match="S_nowhere"):
        net.feet("S1", "S_nowhere")
    with pytest.raises(ArtifactError, match="D_nowhere"):
        net.require_stops("D", "D_nowhere")


@pytest.mark.parametrize("damage", ["diagonal", "axes", "gap"])
def test_distance_table_refuses_damaged_files(tmp_path, damage):
    D = distances()
    if damage == "diagonal":
        D.iloc[1, 1] = 3.0
    elif damage == "axes":
        D = D[["S1", "D", "S2", "S3"]]
    else:
        D.iloc[0, 3] = np.nan
    D.to_csv(tmp_path / "d.csv")
    with pytest.raises(ArtifactError):
        load_distance_table(tmp_path / "d.csv")


def test_usable_faces_need_ok_status_a_face_and_a_known_stop(net):
    faces = net.usable_faces("WZPCK", "F")
    assert sorted(faces.index) == ["FF001A", "FF003A", "GG001A"]   # not XY160A, PL01, FF005A, 007, J100A
    s = net.summary()
    assert s["crosswalk_stops_not_in_distance_table"] == 2           # FF005A (S_gone), PL01 (L_9)
    assert s["n_stops"] == 4 and s["n_locations"] == 8


# --------------------------------------------------------------------------
# sim/workload
# --------------------------------------------------------------------------

def test_every_unplaceable_pick_is_dropped_once_with_a_reason(net):
    ev = events([
        ("O1", "U1", "2026-10-05 08:00:00", "FF001A", 1), ("O1", "U1", "2026-10-05 08:01:00", "XY160A", 1),
        ("O1", "U1", "2026-10-05 08:02:00", "ZZ999A", 1), ("O1", "U1", "2026-10-05 08:03:00", "PL01", 1),
        ("O1", "U1", "2026-10-05 08:03:30", "J100A", 1),          # ok stop, but a reserve slot
        ("O1", "U1", "2026-10-05 08:04:00", "FF005A", 1), ("O1", "U1", "2026-10-05 08:05:00", "FF003A", None),
        (None, "U2", "2026-10-05 09:00:00", "FF001A", 1),
        ("O3", "U3", "2026-10-05 10:00:00", "FF001A", 1, "WZGNA", "PL"),                 # pallet: out of scope
        ("O4", "U4", "2026-10-05 11:00:00", "FF001A", 1, "WZPCK", "CS", "putaway"),     # not a pick
    ])
    wl = build_workload(ev, net, SCOPE, TOUR, START, END)
    # first reason that applies: PL01's crosswalk status (ok_provisional) comes before the face check
    assert wl.counts["dropped_by_reason"] == {"check_snap": 1, "not_in_crosswalk": 1, "ok_provisional": 1,
                                              "not_pick_face": 1, "stop_not_in_distance_table": 1,
                                              "missing_qty": 1, "missing_tour_key": 1}
    assert wl.counts["picks_dropped"] == len(wl.dropped) == 7 and wl.dropped["reason"].notna().all()
    assert wl.counts["picks_kept"] == 1
    assert sum(wl.counts["out_of_scope_in_window"].values()) == 2


def test_leg_after_a_dropped_pick_is_not_merged(net):
    """The leg into FF003A follows a dropped pick: no observed time. The leg
    after it is a real observed leg."""
    ev = events([("O1", "U1", "2026-10-05 08:00:00", "FF001A", 1),
                 ("O1", "U1", "2026-10-05 08:01:00", "XY160A", 1),
                 ("O1", "U1", "2026-10-05 08:02:00", "FF003A", 1),
                 ("O1", "U1", "2026-10-05 08:03:30", "GG001A", 1)])
    (t,) = build_workload(ev, net, SCOPE, TOUR, START, END).tours
    assert [p.location for p in t.picks] == ["FF001A", "FF003A", "GG001A"]
    assert [p.obs_leg_s for p in t.picks] == [None, None, 90.0]
    assert t.n_dropped == 1 and not t.complete


def test_tours_belong_to_the_window_by_first_pick_and_stay_whole(net):
    ev = events([("O1", "U1", "2026-10-04 23:59:00", "FF001A", 1),     # started before: out whole
                 ("O1", "U1", "2026-10-05 00:01:00", "FF003A", 1),
                 ("O2", "U2", "2026-10-11 23:58:00", "FF001A", 1),     # runs past the end: kept whole
                 ("O2", "U2", "2026-10-12 00:02:00", "FF003A", 1)])
    wl = build_workload(ev, net, SCOPE, TOUR, START, END)
    assert [t.order_id for t in wl.tours] == ["O2"] and len(wl.tours[0].picks) == 2
    assert wl.counts["tours_started_before_window"] == 1
    assert wl.counts["tours_running_past_window_end"] == 1


def test_every_tour_is_simulated_whatever_its_size(net):
    ev = events([("O1", "U1", "2026-10-05 08:00:00", "FF001A", 1),
                 ("O2", "U2", "2026-10-05 09:00:00", "FF001A", 1),
                 ("O2", "U2", "2026-10-05 09:01:00", "FF003A", 1)])
    wl = build_workload(ev, net, SCOPE, TOUR, START, END)
    assert len(wl.tours) == 2 and wl.counts["tours_single_pick"] == 1


def test_same_second_scans_follow_the_wms_transaction_number(net):
    ev = events([("O1", "U1", "2026-10-05 08:00:00", "GG001A", 1),
                 ("O1", "U1", "2026-10-05 08:00:00", "FF001A", 1)])
    ev = ev.iloc[::-1]                                   # input order must not matter
    (t,) = build_workload(ev, net, SCOPE, TOUR, START, END).tours
    assert [p.event_id for p in t.picks] == [1, 2]


def test_tour_with_every_pick_dropped_is_counted_not_simulated(net):
    ev = events([("O1", "U1", "2026-10-05 08:00:00", "ZZ999A", 1),
                 ("O1", "U1", "2026-10-05 08:01:00", "XY160A", 1)])
    wl = build_workload(ev, net, SCOPE, TOUR, START, END)
    assert wl.tours == [] and wl.counts["tours_all_picks_dropped"] == 1 and wl.counts["picks_dropped"] == 2


def test_overlapping_tours_of_one_operator_are_counted(net):
    ev = events([("O1", "U1", "2026-10-05 08:00:00", "FF001A", 1),
                 ("O1", "U1", "2026-10-05 08:10:00", "FF003A", 1),
                 ("O2", "U1", "2026-10-05 08:05:00", "GG001A", 1)])
    assert build_workload(ev, net, SCOPE, TOUR, START, END).counts["tours_overlapping_same_user"] == 1


# --------------------------------------------------------------------------
# sim/policy
# --------------------------------------------------------------------------

def test_static_moves_nothing():
    p = make_policy("static")
    assert p.location_for("FF001A", 123.0) == "FF001A" and p.moves() == []
    with pytest.raises(ValueError, match="unknown policy"):
        make_policy("one-shot")


# --------------------------------------------------------------------------
# sim/engine
# --------------------------------------------------------------------------

def test_busy_time_is_the_predicted_legs_and_dispatch_legs_are_feet_only(net):
    """FF001A -> FF003A: 15 ft, same aisle, qty 1 -> 10 + 2 + 1.5 = 13.5 s.
    FF003A -> GG001A: 20 ft, aisle change, qty 3 -> 10 + 6 + 30 + 2 = 48 s.
    Busy 61.5 s. Dispatch D->S1 10 ft + S3->D 40 ft = 50 ft, no seconds."""
    r = sim([tour("FF001A", "FF003A", "GG001A", qtys=[2, 1, 3], arrival=100.0)], net)
    t = r.tours.iloc[0]
    assert t["start_s"] == 100.0 and t["busy_s"] == pytest.approx(61.5) and t["end_s"] == pytest.approx(161.5)
    assert t["pick_feet"] == 35 and t["dispatch_feet"] == 50
    assert (t["first_location"], t["first_stop"]) == ("FF001A", "S1")
    assert r.legs["sim_s"].tolist() == pytest.approx([13.5, 48.0])
    assert r.legs["cross_aisle"].tolist() == [False, True]


def test_operator_pool_limits_concurrent_tours(net):
    class Fixed:
        name = "fixed"

        def predict(self, leg):
            return 50.0
    tours = [tour("FF001A", "FF003A", tid=i) for i in range(3)]
    one = sim(tours, net, operators=1, model=Fixed()).tours
    assert one["start_s"].tolist() == [0, 50, 100] and one["operators_busy_at_start"].max() == 1
    three = sim(tours, net, operators=3, model=Fixed()).tours
    assert (three["wait_s"] == 0).all() and three["operators_busy_at_start"].max() == 3


@pytest.mark.parametrize("key,value", [("dispatch_legs", "timed"), ("drop_off_mode", "real"),
                                       ("start_stop", "D_nowhere")])
def test_trip_ends_outside_the_decision_are_refused(net, key, value):
    with pytest.raises((ValueError, ArtifactError)):
        sim([tour("FF001A", "FF003A")], net, ends={**ENDS, key: value})


def test_moves_are_never_free(net):
    class Mover(StaticPolicy):
        name = "mover"

        def moves(self):
            return [Move(0.0, "FF001A", "FF003A", 12.0)]
    with pytest.raises(ValueError, match="never free"):
        sim([tour("FF001A", "FF003A")], net, policy=Mover())
    r = sim([tour("FF001A", "FF003A")], net, policy=Mover(), move_cost=lambda m: 0.17 * sum(x.cases for x in m))
    assert r.move_cost == pytest.approx(2.04) and len(r.moves) == 1


def test_policy_cannot_place_a_product_off_the_network(net):
    class Bad(StaticPolicy):
        name = "bad"

        def location_for(self, location, t):
            return "XY160A"                         # check_snap: not a usable face
    with pytest.raises(ValueError, match="not a usable pick face"):
        sim([tour("FF001A", "FF003A")], net, policy=Bad())


def test_negative_leg_time_is_refused(net):
    class Broken:
        name = "broken"

        def predict(self, leg):
            return -1.0
    with pytest.raises(ValueError, match="returned"):
        sim([tour("FF001A", "FF003A")], net, model=Broken())


def test_observed_time_is_dropped_where_a_policy_moved_a_product(net):
    class Swap(StaticPolicy):
        name = "swap"

        def location_for(self, location, t):
            return {"FF003A": "GG001A", "GG001A": "FF003A"}.get(location, location)

        def moves(self):
            return [Move(0.0, "FF003A", "GG001A", 1.0)]
    t = Tour(0, "O0", "U0", 0.0, (Pick(1, START, "FF001A", 1.0, None), Pick(2, START, "FF003A", 1.0, 60.0)), 0)
    static = sim([t], net).legs
    swapped = sim([t], net, policy=Swap(), move_cost=lambda m: 0.0).legs
    assert static["obs_s"].tolist() == [60.0] and swapped["obs_s"].isna().all()
    assert swapped["to_location"].tolist() == ["GG001A"]


def test_runs_are_deterministic(net):
    tours = [tour("FF001A", "GG001A", "FF003A", tid=i, arrival=float(i % 2)) for i in range(5)]
    a, b = sim(tours, net, operators=2), sim(tours, net, operators=2)
    pd.testing.assert_frame_equal(a.tours, b.tours)
    pd.testing.assert_frame_equal(a.legs, b.legs)


# --------------------------------------------------------------------------
# leg-model check
# --------------------------------------------------------------------------

def test_check_compares_only_comparable_legs_and_tours(net):
    ev = events([
        # O1: complete, 3 picks, observed legs 60 s and 90 s -> both in the primary set
        ("O1", "U1", "2026-10-05 08:00:00", "FF001A", 1), ("O1", "U1", "2026-10-05 08:01:00", "FF003A", 1),
        ("O1", "U1", "2026-10-05 08:02:30", "GG001A", 1),
        # O2: 2 picks -> below min_picks, excluded from both checks
        ("O2", "U2", "2026-10-05 09:00:00", "FF001A", 1), ("O2", "U2", "2026-10-05 09:01:00", "FF003A", 1),
        # O3: 3 in-scope picks but one dropped -> counts toward min_picks; the leg after the
        # dropped pick has no observed time; the 0 s leg is below the 1 s bound; not complete
        ("O3", "U3", "2026-10-05 10:00:00", "FF001A", 1), ("O3", "U3", "2026-10-05 10:01:00", "XY160A", 1),
        ("O3", "U3", "2026-10-05 10:02:00", "FF003A", 1), ("O3", "U3", "2026-10-05 10:02:00", "GG001A", 1),
    ])
    out = E.run(cfg(), ev, net, Synthetic())
    chk = out.extra["leg_model_check"]
    p, s = chk["primary_pick_to_pick_legs"], chk["secondary_first_to_last_scan"]
    assert p["n_legs"] == 2 and p["median_obs_s"] == 75.0
    # simulated: 13.5 s (FF001A->FF003A) and 48 s (FF003A->GG001A, aisle change, 20 ft, qty 1 -> 10+2+30+2)
    assert p["mae_s"] == pytest.approx(((60 - 13.5) + (90 - 44.0)) / 2)
    assert s["n_tours"] == 1 and s["median_obs_span_s"] == 150.0
    assert chk["what"].startswith("out-of-time check of the leg model") and chk["source"] == "scan_interval_proxy"


def test_check_handles_nothing_to_compare(net):
    ev = events([("O1", "U1", "2026-10-05 08:00:00", "FF001A", 1)])
    chk = E.run(cfg(), ev, net, Synthetic()).extra["leg_model_check"]
    assert chk["primary_pick_to_pick_legs"] == {"n_legs": 0, "n_tours": 0}
    assert chk["secondary_first_to_last_scan"] == {"n_tours": 0}


# --------------------------------------------------------------------------
# run + manifest + script
# --------------------------------------------------------------------------

def test_manifest_records_every_required_field(net, tmp_path):
    ev = events([("O1", "U1", "2026-10-05 08:00:00", "FF001A", 1),
                 ("O1", "U1", "2026-10-05 08:01:00", "FF003A", 1)])
    out = E.run(cfg(), ev, net, Synthetic())
    m = json.loads(json.dumps(write_manifest(tmp_path, cfg(), extra=out.extra), default=str))
    for key in ("config_hash", "git_sha", "seed", "window", "operators", "policy", "leg_time_model",
                "dispatch_legs", "drop_off_mode", "workload", "leg_model_check", "not_findings"):
        assert key in m, key
    assert m["window"] == {"start": "2026-10-05", "end": "2026-10-11", "days": 7}
    assert m["dispatch_legs"] == "untimed" and m["drop_off_mode"] == "provisional"
    assert m["operators"] == 6 and m["leg_time_model"] == "synthetic"
    assert "not findings" in m["not_findings"]["note"]


def test_no_adopted_leg_model_means_no_run():
    with pytest.raises(ValueError, match="no leg-time model has been adopted"):
        E.make_leg_model(cfg())
    with pytest.raises(ValueError, match="unknown leg_time.model"):
        E.make_leg_model(cfg(leg_time={"model": "not_registered"}))
    # one model is registered, and it loads only if its fit was adopted (tests/test_legtime.py)
    assert set(E.LEG_TIME_MODELS) == {"network_feet"}


def test_window_dates_are_inclusive():
    assert E.window_bounds(cfg()) == (START, END)


def test_run_sim_refuses_before_reading_any_picks(tmp_path):
    c = cfg(data={"picks_path": str(tmp_path / "never_read.csv"), "ts_format": "%m/%d/%Y %I:%M:%S %p"},
            travel={"dir": str(tmp_path)}, paths={"out": str(tmp_path / "runs")})
    (tmp_path / "sim.yaml").write_text(yaml.safe_dump(c))
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "run_sim.py"), str(tmp_path / "sim.yaml")],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode != 0 and "no leg-time model has been adopted" in r.stderr
    assert not (tmp_path / "runs").exists()


def test_run_sim_end_to_end_on_a_synthetic_transactions_export(travel_dir, tmp_path, monkeypatch):
    """The converted transactions format (TRANS_TYPE + unpadded TYPE), through
    load_wms_picks, the engine and the manifest, with a synthetic leg model."""
    rows = [("PICK", "9", "10/05/2026 08:00:00 AM", "FF001A"), ("PICK", "9", "10/05/2026 08:01:00 AM", "FF003A"),
            ("PICK", "9", "10/05/2026 08:02:30 AM", "GG001A"), ("DROPOFF-CP", "40", "10/05/2026 08:05:00 AM", "FF001A"),
            ("PICK", "9", "10/05/2026 08:03:00 AM", "XY160A")]
    pd.DataFrame([{"NO": i, "TYPE": code, "TRANS_TYPE": tt, "CREATE_DT": ts, "USER_NAME": "U1",
                   "FROM_ITEM": "1", "FROM_LOCATION": loc, "TO_LOCATION": "V-1", "FROM_QTY": 1,
                   "FROM_UOM": "CS", "ORD_NO": "8006855902", "CUST_NAME": "X", "FROM_ZONE": "WZPCK",
                   "TO_ZONE": "WZEQP"} for i, (tt, code, ts, loc) in enumerate(rows, start=1)]
                 ).to_csv(tmp_path / "tx.csv", index=False)
    c = cfg(leg_time={"model": "synthetic"},
            data={"picks_path": str(tmp_path / "tx.csv"), "ts_format": "%m/%d/%Y %I:%M:%S %p"},
            travel={"dir": str(travel_dir)}, paths={"out": str(tmp_path / "runs")})
    (tmp_path / "sim.yaml").write_text(yaml.safe_dump(c))
    monkeypatch.setitem(E.LEG_TIME_MODELS, "synthetic", Synthetic)
    monkeypatch.setattr(sys, "argv", ["run_sim.py", str(tmp_path / "sim.yaml")])
    runpy.run_path(str(REPO / "scripts" / "run_sim.py"), run_name="__main__")

    run_dir = tmp_path / "runs" / "sim_static_synthetic_20261005_20261011"
    assert sorted(p.name for p in run_dir.iterdir()) == ["dropped_picks.csv", "legs.csv",
                                                         "manifest.json", "tours.csv"]
    m = json.loads((run_dir / "manifest.json").read_text())
    assert m["workload"]["picks_kept"] == 3 and m["workload"]["dropped_by_reason"] == {"check_snap": 1}
    assert len(m["inputs"]) == 3 and all(i["sha256_16"] for i in m["inputs"])
    assert m["headline"]["operator_hours_saved"] == 0 and m["headline"]["labor_rate_source"] == "test rate"
    assert pd.read_csv(run_dir / "dropped_picks.csv")["from_loc"].tolist() == ["XY160A"]
