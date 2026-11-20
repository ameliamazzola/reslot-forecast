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
    return G.fit_profile(load_wms_picks(p, TG.FMT), TG.SCOPE, TG.PLAN_START, TG.PLAN_END, TG.CLUSTER_CFG)


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
    assert m1["totals"] == m2["totals"] and m1["totals"]["orders_arrived"] > 0
    with pytest.raises(SystemExit, match="127.0.0.1 only"):
        run_script("--host", "0.0.0.0")
