"""Visit order for generated tours (src/reslot/sim/routing.py), its calibration
against real tours, and the scripts that use it. Synthetic data only."""
import json
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pytest
import yaml

import test_generator as TG
import test_racksmart as TR
import test_sim as TS
from reslot.io.wms_picks import load_wms_picks
from reslot.sim import generator as G
from reslot.sim import routing as R
from reslot.sim.workload import Pick, Tour
from reslot.travel.artifacts import TravelNetwork

SCOPE = TR.SCOPE
AISLES = {"FF": 100.0, "GG": 200.0, "HH": 300.0}           # aisle x; D sits in front of FF


def grid():
    """Three aisles x five bays; bay k at y = 1000 - 100k (front = bay 0 = largest y);
    feet = |dx| + |dy| between stop points (a test table, not network math)."""
    pts = {"D": (100.0, 1100.0)}
    rows = []
    for a, x in AISLES.items():
        for k in range(5):
            s = f"S_{a}{k}"
            pts[s] = (x, 1000.0 - 100 * k)
            rows.append({"location": f"{a}{k:03d}A", "aisle": a, "zone": "WZPCK", "slot_type": "F", "status": "ok",
                         "stop_id": s, "stop_x_in": str(x), "stop_y_in": str(1000.0 - 100 * k)})
    ids = list(pts)
    D = pd.DataFrame([[abs(pts[a][0] - pts[b][0]) + abs(pts[a][1] - pts[b][1]) for b in ids] for a in ids],
                     index=pd.Index(ids, name="stop_id"), columns=ids)
    return TravelNetwork(D, pd.DataFrame(rows))


@pytest.fixture
def router():
    return R.Router(grid(), SCOPE, "D", pd.Series({"FF001A": 3, "FF004A": 1, "GG002A": 2}))


def test_every_rule_is_a_permutation_and_keeps_a_stop_together(router):
    locs = ["GG002A", "FF004A", "GG002A", "HH001A", "FF001A"]
    for rule in R.RULES:
        o = router.order(locs, rule)
        assert sorted(o) == list(range(len(locs))), rule
        pos = [o.index(i) for i in (0, 2)]
        if rule != "draw":
            assert abs(pos[0] - pos[1]) == 1, rule                         # GG002A twice: one visit
    assert router.order(locs, "draw") == list(range(5))
    with pytest.raises(ValueError, match="unknown routing rule"):
        router.order(locs, "random")


def test_aisle_serpentine_sweeps_from_dispatch_and_alternates_direction(router):
    locs = ["GG002A", "FF004A", "HH001A", "FF001A", "GG004A"]
    order = [locs[i] for i in router.order(locs, "aisle_serpentine")]
    # FF (nearest dispatch) front to back, GG back to front, HH front to back
    assert order == ["FF001A", "FF004A", "GG004A", "GG002A", "HH001A"]


def test_nearest_neighbour_starts_at_dispatch(router):
    locs = ["FF004A", "FF001A", "GG000A"]
    # D->FF001 200 ft ties D->GG000 200 ft: input order wins; then GG000 (200) before FF004 (300)
    assert [locs[i] for i in router.order(locs, "nearest_neighbour")] == ["FF001A", "GG000A", "FF004A"]


def test_work_path_sorts_by_the_wms_path_and_puts_unknown_faces_last(router):
    locs = ["HH003A", "FF001A", "GG002A", "FF004A"]
    assert [locs[i] for i in router.order(locs, "work_path")] == ["FF004A", "GG002A", "FF001A", "HH003A"]
    with pytest.raises(ValueError, match="WORK_PATH"):
        R.Router(grid(), SCOPE, "D").order(locs, "work_path")


def tour(tid, locs):
    ts = pd.Timestamp("2026-09-20 08:00")
    return Tour(tid, str(tid), "U1", 0.0, tuple(Pick(j, ts, l, 1.0, None) for j, l in enumerate(locs)), 0)


def test_calibration_picks_the_rule_that_reproduces_the_scan_order():
    net = grid()
    router = R.Router(net, SCOPE, "D", pd.Series({f"{a}{k:03d}A": -i for i, (a, k) in
                                                  enumerate((a, k) for a in AISLES for k in range(5))}))
    # operators walk serpentine: FF front to back, then GG back to front
    tours = [tour(i, ["FF001A", "FF004A", "GG004A", "GG002A"]) for i in range(20)] + [tour(99, ["HH002A"])]
    res = R.calibrate(tours, router)
    fp = res["feet_per_pick"]
    assert res["tours"] == 20 and res["picks"] == 80                           # single-pick tours left out
    assert res["chosen"] == "aisle_serpentine" and fp["aisle_serpentine"] == fp["actual_scan_order"]
    assert fp["nearest_neighbour"] != fp["actual_scan_order"] and "draw" not in res["candidates"]
    assert res["median_spearman_with_scan_order"]["aisle_serpentine"] == pytest.approx(1.0)
    assert R.calibrate(tours, router) == res                                   # the draw baseline is seeded


def test_resolve_rule_reads_the_calibration_and_refuses_one_past_the_run_start(tmp_path):
    p = tmp_path / "cal.json"
    p.write_text(json.dumps({"chosen": "work_path", "window": ["2026-09-12", "2026-10-12"], "feet_per_pick": {}}))
    rule, info = R.resolve_rule({"rule": "calibrated", "calibration_path": str(p)}, pd.Timestamp("2026-10-12"))
    assert rule == "work_path" and info["how"] == "calibrated"
    with pytest.raises(ValueError, match="after the run start"):
        R.resolve_rule({"rule": "calibrated", "calibration_path": str(p)}, pd.Timestamp("2026-10-11"))
    with pytest.raises(ValueError, match="calibrate_routing.py"):
        R.resolve_rule({"rule": "calibrated", "calibration_path": str(tmp_path / "none.json")})
    with pytest.raises(ValueError, match="generator.routing.rule"):
        R.resolve_rule({"rule": "zigzag"})
    assert R.resolve_rule(None) == ("draw", {"rule": "draw", "how": "set in config"})


def test_generated_tours_are_routed_once_in_the_source(tmp_path):
    net = TravelNetwork(*TR.network())
    p = tmp_path / "plan.csv"
    TG.pick_rows(start=TG.PLAN_START, days=30, n_orders=900).to_csv(p, index=False)
    prof = G.fit_profile(load_wms_picks(p, TG.FMT), SCOPE, TG.PLAN_START, TG.PLAN_END, TG.CLUSTER_CFG)
    g = TG.gcfg(mode="empirical", orders_per_day=300, routing={"rule": "nearest_neighbour"})
    with pytest.raises(ValueError, match="dispatch stop"):
        G.make_generator_source(g, SCOPE, net, prof, TG.RUN_START, 1)
    src = G.make_generator_source(g, SCOPE, net, prof, TG.RUN_START, 1, dispatch_stop="D")
    router = R.Router(net, SCOPE, "D")
    tours = [t for t in src.orders() if len(t.picks) > 1]
    assert tours and src.routing_rule == "nearest_neighbour" and src.describe()["routing_rule"] == "nearest_neighbour"
    for t in tours:                                   # already in the rule's order: routing again changes nothing
        assert router.order([p.location for p in t.picks], "nearest_neighbour") == list(range(len(t.picks)))
    # routing reorders picks only: the same orders and lines as without it
    same = G.make_generator_source(g, SCOPE, net, prof, TG.RUN_START, 1, dispatch_stop="D")
    assert [(t.order_id, sorted(p.location for p in t.picks)) for t in same.orders()] == \
           [(t.order_id, sorted(p.location for p in t.picks)) for t in
            G.make_generator_source({**g, "routing": {"rule": "draw"}}, SCOPE, net, prof, TG.RUN_START, 1).orders()]


def _cfg(tmp_path):
    D, xw = TR.network()
    (tmp_path / "travel").mkdir()
    D.to_csv(tmp_path / "travel" / "stop_distance_ft.csv")
    xw.to_csv(tmp_path / "travel" / "location_stop_crosswalk.csv", index=False)
    TG.pick_rows(start=TG.PLAN_START, days=30, n_orders=900).to_csv(tmp_path / "plan.csv", index=False)
    cfg = yaml.safe_load((TR.REPO / "configs" / "sim.yaml").read_text())
    cfg["travel"]["dir"] = str(tmp_path / "travel")
    cfg["trip_ends"].update(start_stop="D", end_stop="D")
    cfg["paths"]["out"] = str(tmp_path / "out")
    cfg["leg_time"]["model"] = "synthetic"
    cfg["racksmart"].update(planning_picks=[str(tmp_path / "plan.csv")], source_checks=None)
    cfg["generator"]["routing"].update(location_master=str(TR.location_master(tmp_path)),
                                       calibration_path=str(tmp_path / "cal" / "routing_calibration.json"))
    return cfg


def test_calibrate_routing_script_writes_the_choice_and_a_manifest(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    (tmp_path / "sim.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(sys, "argv", ["calibrate_routing.py", str(tmp_path / "sim.yaml")])
    runpy.run_path(str(TR.REPO / "scripts" / "calibrate_routing.py"), run_name="__main__")
    res = json.loads((tmp_path / "cal" / "routing_calibration.json").read_text())
    assert res["chosen"] in R.CANDIDATES and res["window"] == ["2026-09-12 00:00:00", "2026-10-12 00:00:00"]
    assert res["tours"] > 0 and set(res["feet_per_pick"]) == {"actual_scan_order", *R.RULES}
    m = json.loads((tmp_path / "cal" / "manifest.json").read_text())
    assert m["routing_calibration"]["chosen"] == res["chosen"] and "2026-10-12" in m["lookahead"]
    assert R.resolve_rule(cfg["generator"]["routing"], pd.Timestamp("2026-10-12"))[0] == res["chosen"]


def test_live_check_reports_feet_and_seconds_per_pick_for_both_layouts(tmp_path, monkeypatch):
    from reslot.sim import engine as E
    from reslot.sim.racksmart import face_items
    cfg = _cfg(tmp_path)
    ev = load_wms_picks(tmp_path / "plan.csv", TG.FMT)
    prof = G.fit_profile(ev, SCOPE, TG.PLAN_START, TG.PLAN_END, TG.CLUSTER_CFG)
    prof["face_items"] = {str(k): str(v) for k, v in face_items(ev, SCOPE, TG.PLAN_END).items()}
    (tmp_path / "profile.json").write_text(json.dumps(prof))
    cfg["generator"].update(profile_path=str(tmp_path / "profile.json"), orders_per_day=300)
    cfg["generator"]["routing"]["rule"] = "nearest_neighbour"
    (tmp_path / "sim.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setitem(E.LEG_TIME_MODELS, "synthetic", TS.Synthetic)
    monkeypatch.setattr(sys, "argv", ["live_check.py", str(tmp_path / "sim.yaml"), "--hours", "4"])
    runpy.run_path(str(TR.REPO / "scripts" / "live_check.py"), run_name="__main__")
    m = json.loads(next((tmp_path / "out").glob("live_check_*/manifest.json")).read_text())
    rows = {r["side"]: r for r in m["rows"]}
    assert set(rows) == {"live, layout real", "live, layout synthetic"} and m["citable"] is False
    for r in rows.values():
        assert r["routing"] == "nearest_neighbour" and r["picks"] > 0 and r["ft_per_pick"] > 0 and r["s_per_pick"] > 0
