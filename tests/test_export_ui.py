"""Tests for the UI bundle (src/reslot/sim/export.py, scripts/export_ui.py).
Synthetic data only: the aisle from test_racksmart, two runs (static and
racksmart) through scripts/run_sim.py with a synthetic leg model."""
import json
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import pytest
import yaml

import test_racksmart as T
from reslot.sim import engine as E
from reslot.sim.export import BundleError, build_bundle
from reslot.travel.artifacts import CROSSWALK_FILE, DISTANCE_FILE, load_travel
from reslot.utils.manifest import file_hash

REPO = Path(__file__).resolve().parents[1]


class Synthetic:
    name = "synthetic"

    def __init__(self, cfg=None):
        pass

    def predict(self, leg):
        return 20 + 5 * leg.qty + 0.05 * leg.feet


def _run_script(path, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_path(str(REPO / "scripts" / path), run_name="__main__")


@pytest.fixture
def two_runs(tmp_path, monkeypatch):
    travel = tmp_path / "travel"
    travel.mkdir()
    D, xw = T.network()
    D.to_csv(travel / DISTANCE_FILE)
    xw.to_csv(travel / CROSSWALK_FILE, index=False)
    pd.DataFrame({"stop_id": T.STOPS, "x_in": 180855.63, "y_in": [5120.0] + [5000 - 120 * k for k in range(10)],
                  "node_type": ["dispatch"] + ["rack"] * 10, "label": None}).to_csv(travel / "stops.csv", index=False)
    T._pick_rows(T.PLAN, "2026-08-24").assign(CASES_PER_PALLET=60).to_csv(tmp_path / "plan.csv", index=False)
    week = pd.concat([T._pick_rows([([T.A, T.B, T.HOT], 10), ([T.A, T.B, T.QUIET], 2)], f"2026-10-{5 + d:02d}",
                                   transactions=True) for d in range(7)], ignore_index=True)
    week["NO"] = range(1, len(week) + 1)
    week["ORD_NO"] = [f"9{i // 3:06d}" for i in range(len(week))]
    week.to_csv(tmp_path / "tx.csv", index=False)
    T.inventory().to_csv(tmp_path / "inv.csv", index=False)
    monkeypatch.setitem(E.LEG_TIME_MODELS, "synthetic", Synthetic)
    cfg = yaml.safe_load((REPO / "configs" / "sim.yaml").read_text())
    cfg["leg_time"]["model"] = "synthetic"
    cfg["data"]["picks_path"] = str(tmp_path / "tx.csv")
    cfg["travel"]["dir"] = str(travel)
    cfg["trip_ends"].update(start_stop="D", end_stop="D")
    cfg["paths"]["out"] = str(tmp_path / "runs")
    cfg["window"] = {"start": "2026-10-05", "end": "2026-10-11"}      # the synthetic week, not the repo's window
    cfg["racksmart"].update(notebook=str(T.NOTEBOOK), planning_picks=[str(tmp_path / "plan.csv")],
                            inventory_path=str(tmp_path / "inv.csv"), clusters_path=None)
    for pol in ("static", "racksmart"):
        cfg["policy"] = pol
        (tmp_path / f"{pol}.yaml").write_text(yaml.safe_dump(cfg))
        _run_script("run_sim.py", ["run_sim.py", str(tmp_path / f"{pol}.yaml")], monkeypatch)
    runs = {p: tmp_path / "runs" / f"sim_{p}_synthetic_20261005_20261011" for p in ("static", "racksmart")}
    return tmp_path, travel, runs


def _bundle(travel, run_dirs):
    stops = pd.read_csv(travel / "stops.csv", dtype={"stop_id": str, "label": str})
    hashes = {f: file_hash(travel / f) for f in (DISTANCE_FILE, CROSSWALK_FILE)}
    return build_bundle([str(d) for d in run_dirs], load_travel(travel), stops, "2026-11-03T00:00:00", hashes)


def test_bundle_regroups_the_runs_own_outputs(two_runs):
    _, travel, runs = two_runs
    b = _bundle(travel, runs.values())
    assert b["schema"] == "racksmart-ui/1" and len(b["dates"]) == 7
    for r in b["runs"]:
        kept = r["manifest"]["workload"]["picks_kept"]
        assert sum(r["daily"]["picks"]) == kept == 7 * 12 * 3
        assert sum(sum(d.values()) for d in r["stop_picks"]) == kept       # every pick has a stop
        assert r["daily"]["picks"] == [36.0] * 7
        assert r["manifest"]["dispatch_legs"] == "untimed" and "config" not in r["manifest"]
    rs = next(r for r in b["runs"] if r["policy"] == "racksmart")
    assert [(s["location_a"], s["stop_a"], s["location_b"], s["stop_b"]) for s in rs["swaps"]] in (
        [(T.HOT, "S9", T.QUIET, "S2")], [(T.QUIET, "S2", T.HOT, "S9")])
    assert rs["swaps"][0]["picks"] == 7 * (10 + 2)        # every pick of both swapped products, all week
    rack = {s["id"]: s for s in b["network"]["stops"] if s["type"] == "rack"}
    assert rack["S9"]["faces"] == 1 and rack["S9"]["ft_from_dispatch"] == 100 and rack["S9"]["aisle"] == "FF"


def test_export_script_writes_a_bundle(two_runs, monkeypatch):
    tmp, travel, runs = two_runs
    out = tmp / "bundle.json"
    _run_script("export_ui.py", ["export_ui.py", *map(str, runs.values()), "--travel", str(travel), "--out", str(out)],
                monkeypatch)
    b = json.loads(out.read_text())
    assert sorted(r["policy"] for r in b["runs"]) == ["racksmart", "static"]


def test_runs_from_another_week_are_refused(two_runs):
    _, travel, runs = two_runs
    m = runs["racksmart"] / "manifest.json"
    j = json.loads(m.read_text())
    j["window"]["start"] = "2026-09-28"
    m.write_text(json.dumps(j))
    with pytest.raises(BundleError, match="window"):
        _bundle(travel, runs.values())


def test_runs_over_different_picks_are_refused(two_runs):
    _, travel, runs = two_runs
    m = runs["racksmart"] / "manifest.json"
    j = json.loads(m.read_text())
    j["inputs"][0]["sha256_16"] = "0000000000000000"
    m.write_text(json.dumps(j))
    with pytest.raises(BundleError, match="different picks file"):
        _bundle(travel, runs.values())


def test_runs_on_another_network_are_refused(two_runs):
    _, travel, runs = two_runs
    xw = pd.read_csv(travel / CROSSWALK_FILE, dtype=str)
    xw.loc[0, "status"] = "check_snap"                     # the network changed after the runs
    xw.to_csv(travel / CROSSWALK_FILE, index=False)
    with pytest.raises(BundleError, match="different location_stop_crosswalk.csv"):
        _bundle(travel, runs.values())


def test_same_policy_twice_and_old_runs_are_refused(two_runs):
    _, travel, runs = two_runs
    with pytest.raises(BundleError, match="same policy"):
        _bundle(travel, [runs["static"], runs["static"]])
    t = pd.read_csv(runs["static"] / "tours.csv").drop(columns=["first_stop"])
    t.to_csv(runs["static"] / "tours.csv", index=False)
    with pytest.raises(BundleError, match="first_stop"):
        _bundle(travel, [runs["static"]])


def test_golden_zone_reaches_the_bundle_only_for_methods_that_use_it(two_runs):
    _, travel, runs = two_runs
    assert (runs["racksmart"] / "golden_faces.csv").exists()            # written for every notebook method
    b = _bundle(travel, runs.values())
    rs = next(r for r in b["runs"] if r["policy"] == "racksmart")
    assert rs["golden_stops"] == []                                      # RackSmart does not swap into the zone
    m = runs["racksmart"] / "manifest.json"
    j = json.loads(m.read_text())
    j["notebook_plan"]["uses_golden_zone"] = True                         # as the ABC methods record it
    m.write_text(json.dumps(j))
    rs = next(r for r in _bundle(travel, runs.values())["runs"] if r["policy"] == "racksmart")
    # the plan's faces are every structural face (all ten), picked or not; the seven within
    # 60 ft of the aisle front are golden
    assert rs["golden_stops"] == [f"S{k}" for k in range(7)]


def test_abc_methods_record_that_they_use_the_golden_zone():
    from reslot.sim import racksmart as RS
    assert {k for k, v in RS.METHODS.items() if v["uses_golden_zone"]} == {"classic_abc", "abc_cost_test"}


# --------------------------------------------------------------------------
# scripts/run_all.py: sim for every policy -> bundle, in one command
# --------------------------------------------------------------------------

def _run_all_setup(tmp_path, outcome="adopted"):
    import subprocess  # noqa: F401
    import test_legtime as LT
    travel = tmp_path / "travel"
    travel.mkdir()
    D, xw = T.network()
    D.to_csv(travel / DISTANCE_FILE)
    xw.to_csv(travel / CROSSWALK_FILE, index=False)
    pd.DataFrame({"stop_id": T.STOPS, "x_in": 180855.63, "y_in": [5120.0] + [5000 - 120 * k for k in range(10)],
                  "node_type": ["dispatch"] + ["rack"] * 10, "label": None}).to_csv(travel / "stops.csv", index=False)
    T._pick_rows(T.PLAN, "2026-08-24").assign(CASES_PER_PALLET=60).to_csv(tmp_path / "plan.csv", index=False)
    week = pd.concat([T._pick_rows([([T.A, T.B, T.HOT], 6), ([T.A, T.B, T.QUIET], 2)], f"2026-10-{5 + d:02d}",
                                   transactions=True) for d in range(7)], ignore_index=True)
    week["NO"] = range(1, len(week) + 1)
    week["ORD_NO"] = [f"9{i // 3:06d}" for i in range(len(week))]
    week.to_csv(tmp_path / "tx.csv", index=False)
    T.inventory().to_csv(tmp_path / "inv.csv", index=False)
    (tmp_path / "fit").mkdir()
    lt = LT.write_fit(tmp_path / "fit", travel, outcome=outcome)
    cfg = yaml.safe_load((REPO / "configs" / "sim.yaml").read_text())
    cfg["leg_time"].update(lt["leg_time"])
    cfg["data"]["picks_path"] = str(tmp_path / "tx.csv")
    cfg["travel"]["dir"] = str(travel)
    cfg["trip_ends"].update(start_stop="D", end_stop="D")
    cfg["paths"]["out"] = str(tmp_path / "runs")
    cfg["window"] = {"start": "2026-10-05", "end": "2026-10-11"}
    cfg["racksmart"].update(notebook=str(T.NOTEBOOK), planning_picks=[str(tmp_path / "plan.csv")],
                            inventory_path=str(tmp_path / "inv.csv"), clusters_path=None)
    (tmp_path / "sim.yaml").write_text(yaml.safe_dump(cfg))
    return tmp_path / "sim.yaml"


def test_run_all_runs_every_policy_and_bundles(tmp_path):
    import subprocess
    cfg = _run_all_setup(tmp_path)
    out = tmp_path / "bundle.json"
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "run_all.py"), "--config", str(cfg), "--out", str(out)],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    b = json.loads(out.read_text())
    # the three UI tabs plus the static reference
    assert sorted(x["policy"] for x in b["runs"]) == ["abc_cost_test", "family_grouping", "racksmart", "static"]
    assert {x["leg_time_model"] for x in b["runs"]} == {"network_feet"}
    assert len({x["manifest"]["config_hash"] for x in b["runs"]}) == 4    # each run records its own policy


def test_run_all_sensitivity_reruns_the_methods_with_pallet_fill_and_compares(tmp_path):
    import subprocess
    cfg = _run_all_setup(tmp_path)
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "run_all.py"), "--config", str(cfg),
                        "--policies", "static", "racksmart", "--sensitivity"], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    runs = tmp_path / "runs"
    main = json.loads((runs / "sim_racksmart_network_feet_20261005_20261011" / "manifest.json").read_text())
    pf = json.loads((runs / "sim_racksmart_network_feet_20261005_20261011_pallet_fill" / "manifest.json").read_text())
    assert main["notebook_plan"]["move_cases"]["basis"] == "inventory_snapshot"
    assert list(main["planning_inputs"]["lookahead_exceptions_used"]) == ["move_cases"]
    assert pf["notebook_plan"]["move_cases"]["basis"] == "pallet_fill"
    assert pf["planning_inputs"]["lookahead_exceptions_used"] == {}
    assert not (runs / "sim_static_network_feet_20261005_20261011_pallet_fill").exists()
    cmp_dir = runs / "compare_snapshot_vs_pallet_fill_20261005_20261011"
    m = json.loads((cmp_dir / "manifest.json").read_text())
    assert set(m["rankings"]) == {"notebook_net", "sim_busy"} and (cmp_dir / "comparison.csv").exists()
    assert "ranking by notebook_net" in r.stdout


def test_run_all_stops_before_simulating_when_the_model_is_not_adopted(tmp_path):
    import subprocess
    cfg = _run_all_setup(tmp_path, outcome="rejected")
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "run_all.py"), "--config", str(cfg)],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode != 0 and "review outcome is 'rejected'" in (r.stdout + r.stderr)
    assert not (tmp_path / "runs").exists()

