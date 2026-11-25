"""Live mode (src/reslot/sim/live.py, scripts/run_live.py): the 127.0.0.1-only
server, controls through the source's control log, the throttled event stream,
the live manifest, and determinism at a fixed sim span. Synthetic data only."""
import http.client
import json
import runpy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import pytest
import yaml

import test_generator as TG
import test_racksmart as TR
import test_sim as TS
from reslot.io.wms_picks import load_wms_picks
from reslot.sim import generator as G
from reslot.sim import live as L
from reslot.sim.policy import make_policy
from reslot.travel.artifacts import TravelNetwork

UI = TR.REPO / "ui"
FAST = 100000.0


@pytest.fixture(scope="module")
def net():
    return TravelNetwork(*TR.network())


@pytest.fixture(scope="module")
def profile(tmp_path_factory):
    p = tmp_path_factory.mktemp("prof") / "plan.csv"
    TG.pick_rows(start=TG.PLAN_START, days=30, n_orders=900).to_csv(p, index=False)
    from reslot.sim.racksmart import face_items
    ev = load_wms_picks(p, TG.FMT)
    prof = G.fit_profile(ev, TG.SCOPE, TG.PLAN_START, TG.PLAN_END, TG.CLUSTER_CFG)
    prof["face_items"] = {str(k): str(v) for k, v in face_items(ev, TG.SCOPE, TG.PLAN_END).items()}
    return prof


def make_run(net, profile, speed=FAST, speeds=(1, 10, 60, 600, 3000, FAST), hours=4.0, seed=7, **g):
    src = G.make_generator_source(TG.gcfg(mode="empirical", seed=seed, orders_per_day=300, **g), TG.SCOPE, net,
                                  profile, TG.RUN_START, 2)
    return L.LiveRun(src, make_policy("static"), net, TS.Synthetic(), TG.SCOPE, dict(TR.ENDS), 3, speed,
                     list(speeds), until_s=hours * 3600)


def post(port, body):
    c = http.client.HTTPConnection(L.LOCAL_HOST, port, timeout=5)
    c.request("POST", "/control", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    return r.status, json.loads(r.read())


def get_json(port, path):
    c = http.client.HTTPConnection(L.LOCAL_HOST, port, timeout=5)
    c.request("GET", path)
    r = c.getresponse()
    return r.status, json.loads(r.read())


# --------------------------------------------------------------------------

def test_server_refuses_any_non_local_bind(net, profile):
    run = make_run(net, profile)
    for host in ("0.0.0.0", "localhost", "192.168.1.5", "::"):
        with pytest.raises(ValueError, match="127.0.0.1 only"):
            L.serve(run, host, 0, UI)


def test_control_changes_speed_and_affinity_and_both_are_logged(net, profile):
    run = make_run(net, profile, speed=1, hours=1)
    server = L.serve(run, L.LOCAL_HOST, 0, UI)
    port = server.server_address[1]
    run.start()
    try:
        time.sleep(0.3)
        code, body = post(port, {"speed": 600})
        assert code == 200 and run.env.speed == 600
        log = run.source.controls
        assert log[-1]["control"] == "speed" and log[-1]["value"] == 600 and log[-1]["previous"] == 1
        assert 0 <= log[-1]["sim_time_s"] < 3600 and body["controls"] == log
        assert post(port, {"speed": 7})[0] == 400                      # not one of the configured speeds
        assert post(port, {"affinity": 0.8})[0] == 200 and run.source.gen.affinity == 0.8
        assert run.source.controls[-1]["control"] == "affinity"
        assert post(port, {"affinity": 2})[0] == 400 and post(port, {})[0] == 400
        assert [s["speed"] for s in run.speeds_used] == [1.0, 600.0]
        time.sleep(0.4)                                  # /state is the latest published snapshot (<= 4 a second)
        code, state = get_json(port, "/state")
        assert code == 200 and state["policy"] == "static" and state["speed"] == 600
        c = http.client.HTTPConnection(L.LOCAL_HOST, port, timeout=5)    # ui/ is served
        c.request("GET", "/index.html")
        assert c.getresponse().status == 200
    finally:
        run.stop()
        server.shutdown()


def test_a_fast_run_streams_events_with_rising_sim_time_at_most_four_a_second(net, profile):
    run = make_run(net, profile, speed=3000, hours=3)                    # ~3.6 s of wall clock
    server = L.serve(run, L.LOCAL_HOST, 0, UI)
    port = server.server_address[1]
    run.start()
    try:
        c = http.client.HTTPConnection(L.LOCAL_HOST, port, timeout=5)
        c.request("GET", "/events")
        r = c.getresponse()
        assert r.status == 200 and r.getheader("Content-Type") == "text/event-stream"
        snaps, t0 = [], time.monotonic()
        while time.monotonic() - t0 < 2.0:
            line = r.fp.readline().decode().strip()
            if line.startswith("data: "):
                snaps.append(json.loads(line[6:]))
        elapsed = time.monotonic() - t0
        c.close()
    finally:
        run.stop()
        server.shutdown()
    assert len(snaps) >= 4 and len(snaps) <= 4 * elapsed + 2           # throttled to 4 a second
    times = [s["sim_time_s"] for s in snaps]
    assert times == sorted(times) and times[-1] > times[0]
    assert [s["seq"] for s in snaps] == sorted(s["seq"] for s in snaps)
    s = snaps[-1]
    for k in ("sim_clock", "orders_arrived", "orders_in_progress", "orders_done", "queue_length",
              "operators_busy", "picks_by_stop", "pick_feet", "operator_hours", "policy"):
        assert k in s, k
    assert sum(sum(x["picks_by_stop"].values()) for x in snaps) <= s["picks_done"]


def test_same_seed_no_controls_same_totals(net, profile):
    def totals(seed):
        run = make_run(net, profile, hours=6, seed=seed)
        run.start()
        assert run.wait(60)
        return run.totals()
    a, b = totals(7), totals(7)
    assert a == b and a["orders_done"] > 10 and a["sim_time_s"] == 6 * 3600
    assert totals(8) != a


def test_run_live_writes_a_non_citable_manifest(tmp_path, monkeypatch, net, profile):
    from reslot.sim import engine as E
    D, xw = TR.network()
    (tmp_path / "travel").mkdir()
    D.to_csv(tmp_path / "travel" / "stop_distance_ft.csv")
    xw.to_csv(tmp_path / "travel" / "location_stop_crosswalk.csv", index=False)
    (tmp_path / "profile.json").write_text(json.dumps(profile))
    cfg = yaml.safe_load((TR.REPO / "configs" / "sim.yaml").read_text())
    cfg["leg_time"]["model"] = "synthetic"
    cfg["travel"]["dir"] = str(tmp_path / "travel")
    cfg["trip_ends"].update(start_stop="D", end_stop="D")
    cfg["paths"]["out"] = str(tmp_path / "out")
    cfg["generator"].update(profile_path=str(tmp_path / "profile.json"), orders_per_day=300)
    cfg["generator"]["routing"]["rule"] = "nearest_neighbour"
    cfg["live"].update(speeds=[1, 10, 60, 600, FAST], speed=FAST)
    (tmp_path / "sim.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setitem(E.LEG_TIME_MODELS, "synthetic", TS.Synthetic)

    def run_script(*extra):
        monkeypatch.setattr(sys, "argv", ["run_live.py", str(tmp_path / "sim.yaml"), "--port", "0", "--hours", "3",
                                          *extra])
        runpy.run_path(str(TR.REPO / "scripts" / "run_live.py"), run_name="__main__")

    run_script("--seed", "11")
    run_script("--seed", "11")
    dirs = sorted((tmp_path / "out").glob("live_*"))
    assert len(dirs) == 2
    m1, m2 = (json.loads((d / "manifest.json").read_text()) for d in dirs)
    for m in (m1, m2):
        assert m["mode"] == "live" and m["citable"] is False and m["policy"] == "static" and m["seed"] == 11
        assert m["source"]["type"] == "generator" and m["source"]["controls"] == []
        assert m["sim_time_covered"]["seconds"] == 3 * 3600 and m["speeds_used"] == [{"sim_time_s": 0.0, "speed": FAST}]
        assert m["wall_clock"]["start"] and m["wall_clock"]["end"] and m["planning_inputs"]["inputs"] == {}
        assert m["source"]["mode_label"].startswith("empirical")
        assert m["source"]["routing_rule"] == "nearest_neighbour" and m["source"]["layout"]["sku_to_face"] == "real"
    assert m1["totals"] == m2["totals"] and m1["totals"]["orders_arrived"] > 0
    with pytest.raises(SystemExit, match="127.0.0.1 only"):
        run_script("--host", "0.0.0.0")


def test_live_feet_so_far_is_replays_pick_feet(net, profile):
    """Same quantity: pick-to-pick leg feet within tours, dispatch legs excluded;
    seconds = operator busy time; both per pick with first picks counted."""
    from reslot.sim import engine as E
    run = make_run(net, profile, speed=FAST, hours=48)                 # the whole 2-day window
    run.start()
    assert run.wait(60)
    src = G.make_generator_source(TG.gcfg(mode="empirical", seed=7, orders_per_day=300), TG.SCOPE, net,
                                  profile, TG.RUN_START, 2)
    res = E.simulate(src, 3, make_policy("static"), net, TS.Synthetic(), TG.SCOPE, dict(TR.ENDS))
    t = run.totals()
    assert t["orders_in_progress_at_stop"] == 0 and t["orders_done"] == len(res.tours)
    assert t["pick_feet"] == pytest.approx(res.tours["pick_feet"].sum())
    assert t["picks_done"] == res.tours["n_picks"].sum()
    assert t["operator_hours"] * 3600 == pytest.approx(res.tours["busy_s"].sum())


def test_arrival_cutoff_drains_active_and_queued_tours():
    from types import SimpleNamespace

    cutoff = 8 * 3600
    net = TravelNetwork(TS.distances(), TS.crosswalk())
    tours = [
        TS.tour("FF001A", "FF003A", tid=0, arrival=cutoff - 2),
        TS.tour("FF003A", tid=1, arrival=cutoff - 1),
        TS.tour("FF003A", "GG001A", tid=2, arrival=cutoff),
        TS.tour("GG001A", tid=3, arrival=cutoff + 1),
    ]
    source = SimpleNamespace(start=TS.START, gen=SimpleNamespace(affinity=0), orders=lambda: iter(tours))
    run = L.LiveRun(source, make_policy("static"), net, TS.Synthetic(), TS.SCOPE, TS.ENDS, 1,
                    FAST, [FAST], arrival_cutoff_s=cutoff)
    run.start()
    assert run.wait(10)

    replay = TS.sim(tours, net, operators=1).tours
    admitted = replay[replay["arrival_s"] < cutoff]
    assert admitted["start_s"].iloc[1] > cutoff
    assert admitted["dispatch_feet"].sum() > 0
    totals = run.totals()
    assert totals["orders_arrived"] == totals["orders_done"] == len(admitted) == 2
    assert totals["orders_in_progress_at_stop"] == totals["queue_length_at_stop"] == 0
    assert totals["picks_done"] == admitted["n_picks"].sum() == 3
    assert totals["pick_feet"] == admitted["pick_feet"].sum() == 15
    assert totals["operator_hours"] * 3600 == pytest.approx(admitted["busy_s"].sum())
    assert totals["sim_time_s"] == admitted["end_s"].max() > cutoff


def _cutoff_tours(cutoff):
    """Tour 0 is released 2 s before the cutoff and runs across it; tour 1 is released
    1 s before it and queues behind tour 0 (one operator); tour 2 is released exactly
    at the cutoff and tour 3 after it -- neither may count."""
    return [TS.tour("FF001A", "FF003A", tid=0, arrival=cutoff - 2),
            TS.tour("FF003A", tid=1, arrival=cutoff - 1),
            TS.tour("FF003A", "GG001A", tid=2, arrival=cutoff),
            TS.tour("GG001A", tid=3, arrival=cutoff + 1)]


def test_boundary_rule_counts_a_tour_crossing_the_cutoff_whole_on_both_sides():
    """Arrivals stop at the cutoff, both drain: the tour released before the cutoff
    that finishes after it counts whole in live and in replay; tours released at or
    after the cutoff count on neither side, although replay ran them."""
    from types import SimpleNamespace

    cutoff = 8 * 3600
    net = TravelNetwork(TS.distances(), TS.crosswalk())
    tours = _cutoff_tours(cutoff)
    source = SimpleNamespace(start=TS.START, gen=SimpleNamespace(affinity=0), orders=lambda: iter(tours))
    run = L.LiveRun(source, make_policy("static"), net, TS.Synthetic(), TS.SCOPE, TS.ENDS, 1,
                    FAST, [FAST], arrival_cutoff_s=cutoff)
    run.start()
    assert run.wait(10)
    live = L.live_window_row(run, "live", "as released")

    replay_tours = TS.sim(tours, net, operators=1).tours           # replay ran every tour, past the cutoff too
    assert len(replay_tours) == 4
    crossing = replay_tours.set_index("tour_id").loc[0]
    assert crossing["arrival_s"] < cutoff < crossing["end_s"]         # released before, finished after
    replay = L.replay_window_row(replay_tours, cutoff)

    whole = replay_tours[replay_tours["tour_id"].isin([0, 1])]        # the two tours released before the cutoff
    for row in (live, replay):
        assert row["tours"] == 2 and row["picks"] == whole["n_picks"].sum() == 3
        assert row["feet"] == whole["pick_feet"].sum() == 15            # tour 0's leg counted in full
        assert row["operator_hours"] * 3600 == pytest.approx(whole["busy_s"].sum())
        assert row["drain_end_s"] == whole["end_s"].max() > cutoff
        assert row["ft_per_pick"] == pytest.approx(15 / 3) and row["s_per_pick"] == pytest.approx(whole["busy_s"].sum() / 3)
    assert {k: v for k, v in live.items() if k not in ("side", "routing")} == \
           {k: v for k, v in replay.items() if k not in ("side", "routing")}
    # tours 2 and 3 (released at / after the cutoff) are absent from both rows
    late = replay_tours[replay_tours["tour_id"].isin([2, 3])]
    assert replay["picks"] + late["n_picks"].sum() == replay_tours["n_picks"].sum()


def test_a_live_run_that_did_not_drain_is_not_comparable(net, profile):
    from types import SimpleNamespace

    no_cutoff = make_run(net, profile, hours=2)
    no_cutoff.start()
    assert no_cutoff.wait(60)
    with pytest.raises(ValueError, match="arrival cutoff"):
        L.live_window_row(no_cutoff, "live", "x")
    # stopped by a sim-time limit 1 s before the cutoff: tour 0 is mid-leg, tour 1 queued
    cutoff = 8 * 3600
    tours = _cutoff_tours(cutoff)
    source = SimpleNamespace(start=TS.START, gen=SimpleNamespace(affinity=0), orders=lambda: iter(tours))
    run = L.LiveRun(source, make_policy("static"), TravelNetwork(TS.distances(), TS.crosswalk()), TS.Synthetic(),
                    TS.SCOPE, TS.ENDS, 1, FAST, [FAST], until_s=cutoff - 0.5, arrival_cutoff_s=cutoff)
    run.start()
    assert run.wait(10)
    assert run.totals()["orders_in_progress_at_stop"] == 1
    with pytest.raises(ValueError, match="did not drain"):
        L.live_window_row(run, "live", "x")


def test_boundary_rule_label_names_the_release_window():
    assert L.boundary_rule(pd.Timestamp("2026-10-12"), 8 * 3600) == (
        "tours released 2026-10-12 00:00-08:00 (end exclusive), each run to completion; "
        "tours released at or after the cutoff are not counted")
    assert "2026-10-12 00:00 to 2026-10-13 06:00" in L.boundary_rule(pd.Timestamp("2026-10-12"), 30 * 3600)


def test_a_line_scanned_in_three_rows_is_one_line_in_replay_and_in_the_measurement():
    """Replay counts scan rows. One SKU on one order scanned in three rows at one face
    is one SKU line and one stop visit (two 0-ft legs), and the per-line figures use
    that, not the rows."""
    from reslot.sim.workload import build_workload
    net = TravelNetwork(TS.distances(), TS.crosswalk())
    t0 = TS.START + pd.Timedelta(hours=8)
    rows = [("FF001A", "A1", 0), ("FF001A", "A1", 20), ("FF001A", "A1", 40),   # one line, three scan rows
            ("FF003A", "B2", 90), ("GG001A", "C3", 150)]
    ev = TS.events([("8001", "U1", str(t0 + pd.Timedelta(seconds=s)), loc, 1.0) for loc, _, s in rows])
    ev["item_id"] = pd.array([item for _, item, _ in rows], dtype="string")
    work = build_workload(ev, net, TS.SCOPE, TS.TOUR, TS.START, TS.END)
    (tour,) = work.tours
    assert len(tour.picks) == 5 and tour.n_lines == 3 and [p.item_id for p in tour.picks][:3] == ["A1"] * 3
    res = TS.sim(work.tours, net, operators=1)
    r = res.tours.iloc[0]
    assert (r["n_picks"], r["n_lines"], r["n_stop_visits"]) == (5, 3, 3)
    assert (res.legs["feet"] == 0).sum() == 2                                   # the two repeat rows
    row = L.replay_window_row(res.tours, 24 * 3600)
    assert (row["picks"], row["lines"], row["stop_visits"], row["tours"]) == (5, 3, 3, 1)
    assert row["ft_per_line"] == pytest.approx(row["feet"] / 3) and row["ft_per_pick"] == pytest.approx(row["feet"] / 5)
    assert row["s_per_line"] == pytest.approx(r["busy_s"] / 3)
