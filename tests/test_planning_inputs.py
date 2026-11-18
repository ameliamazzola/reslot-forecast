"""Planning inputs after the lookahead fix: structural faces, the face -> item map,
the two case-count sources, cluster cutoffs, the engine's per-input check, and the
run comparison. Synthetic data only."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "affinity"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import pytest

import test_racksmart as TR
from reslot.io.wms_picks import load_wms_picks
from reslot.schemas import EVENTS, validate
from reslot.sim import compare as C
from reslot.sim import engine as E
from reslot.sim import racksmart as RS
from reslot.sim.policy import StaticPolicy
from reslot.travel.artifacts import TravelNetwork
from reslot.utils.manifest import file_hash

SCOPE = TR.SCOPE
END = pd.Timestamp("2026-09-21")


def picks(rows):
    """rows: (event_id, ts, location, item, cases_per_pallet)"""
    df = pd.DataFrame([{"event_id": e, "ts": pd.Timestamp(t), "event_class": "pick", "from_zone": "WZPCK",
                        "uom": "CS", "from_loc": l, "item_id": i, "qty": 1.0, "order_id": "1", "user_id": "U"}
                       for e, t, l, i, _ in rows])
    ev = validate(df, EVENTS, "events", strict=False)
    ev["cases_per_pallet"] = [c for *_, c in rows]
    return ev


# --------------------------------------------------------------------------
# faces and items
# --------------------------------------------------------------------------

def test_structural_faces_are_the_usable_faces_and_nothing_else():
    D, xw = TR.network()
    xw.loc[3, "status"] = "check_snap"                        # not ok -> not a face
    xw.loc[4, "slot_type"] = "R"                              # reserve -> not a face
    faces = RS.structural_faces(TravelNetwork(D, xw), SCOPE)
    assert faces == sorted(set(TR.LOCS) - {TR.LOCS[3], TR.LOCS[4]})
    assert "location master" in RS.FACE_RULE and "No stock, no picks" in RS.FACE_RULE


def test_face_item_is_the_last_pick_before_the_planning_end():
    ev = picks([(1, "2026-09-01 08:00", "FF001A", "A", 60), (2, "2026-09-10 08:00", "FF001A", "B", 60),
                (3, "2026-09-10 08:00", "FF002A", "C", 60), (4, "2026-09-10 08:00", "FF002A", "D", 60),
                (5, "2026-09-21 00:00", "FF001A", "LATE", 60), (6, "2026-09-25 08:00", "FF003A", "LATE", 60)])
    items = RS.face_items(ev, SCOPE, END)
    assert items.to_dict() == {"FF001A": "B", "FF002A": "D"}            # same time: higher event_id wins


# --------------------------------------------------------------------------
# case counts
# --------------------------------------------------------------------------

def test_pallet_fill_uses_each_items_pallet_size_times_the_fill():
    ev = picks([(1, "2026-09-01", "FF001A", "A", 40), (2, "2026-09-02", "FF001A", "A", 60),
                (3, "2026-09-03", "FF002A", "B", 100), (4, "2026-09-04", "FF003A", "C", -1),
                (5, "2026-09-22", "FF002A", "B", 9999)])                  # after the end: ignored
    items = RS.face_items(ev, SCOPE, END)
    faces = ["FF001A", "FF002A", "FF003A", "FF004A"]
    cases, info = RS.cases_from_pallet_fill(ev, SCOPE, END, items, faces, 0.5)
    # A: median(40, 60) = 50; B: 100; C has no valid size and FF004A no item -> median(50, 100) = 75
    assert cases.to_dict() == {"FF001A": 25.0, "FF002A": 50.0, "FF003A": 37.5, "FF004A": 37.5}
    assert info == {"basis": "pallet_fill", "face_fill": 0.5, "default_cases_per_pallet": 75.0,
                    "faces_with_item_pallet_size": 2}
    cases, _ = RS.cases_from_pallet_fill(ev, SCOPE, END, items, faces, 0.5, default_cpp=20)
    assert cases["FF004A"] == 10.0
    with pytest.raises(ValueError, match="face_fill"):
        RS.cases_from_pallet_fill(ev, SCOPE, END, items, faces, 1.5)


def test_pallet_fill_without_any_pallet_size_needs_a_default():
    ev = picks([(1, "2026-09-01", "FF001A", "A", float("nan"))])
    with pytest.raises(ValueError, match="cases_per_pallet"):
        RS.cases_from_pallet_fill(ev, SCOPE, END, RS.face_items(ev, SCOPE, END), ["FF001A"], 0.75)


def test_inventory_cases_follow_load_york():
    cases, info = RS.cases_from_inventory(TR.inventory(), TR.LOCS)
    assert cases["FF009A"] == 5.0 and info["faces_with_inventory"] == 9 and info["basis"] == "inventory_snapshot"


def test_the_loader_carries_cases_per_pallet(tmp_path):
    rows = TR._pick_rows([([TR.A, TR.B, TR.HOT], 1)], "2026-08-24")
    rows.to_csv(tmp_path / "no_cpp.csv", index=False)
    rows.assign(CASES_PER_PALLET=[60, "", 238]).to_csv(tmp_path / "cpp.csv", index=False)
    assert load_wms_picks(tmp_path / "no_cpp.csv")["cases_per_pallet"].isna().all()
    got = load_wms_picks(tmp_path / "cpp.csv")["cases_per_pallet"].tolist()
    assert got[0] == 60 and pd.isna(got[1]) and got[2] == 238


# --------------------------------------------------------------------------
# clusters: the cutoff travels with the file
# --------------------------------------------------------------------------

def _clusters(tmp_path, cutoff="2026-09-21", sha=None):
    path = tmp_path / "sku_clusters.csv"
    pd.DataFrame({"cluster": [0, 0], "sku": ["1", "20000001"], "descr": "x", "orders": 9}).to_csv(path, index=False)
    from graph_export import write_meta
    write_meta(tmp_path / "sku_clusters_meta.json", cutoff, sku_clusters_sha256_16=sha or file_hash(path))
    return path


def test_clusters_need_a_matching_meta(tmp_path):
    path = _clusters(tmp_path)
    cutoff, meta = RS.cluster_cutoff(path)
    assert cutoff == pd.Timestamp("2026-09-21") and meta["cutoff_exclusive"] == "2026-09-21"
    _clusters(tmp_path, sha="0" * 16)
    with pytest.raises(ValueError, match="different"):
        RS.cluster_cutoff(path)
    (tmp_path / "sku_clusters_meta.json").unlink()
    with pytest.raises(ValueError, match="cutoff is unknown"):
        RS.cluster_cutoff(path)


def _planning(tmp_path):
    import test_no_lookahead as NL
    return NL._planning_cfg(tmp_path)


def test_clusters_place_faces_by_the_notebooks_rule_on_planning_picks(tmp_path):
    """location_clusters (the notebook's): a face takes the cluster of the item
    with the most cases picked there in the planning window -- the adapter only
    supplies the planning picks (extra["sku"]) and the file path."""
    cfg, n = _planning(tmp_path)
    cfg["racksmart"]["clusters_path"] = str(_clusters(tmp_path))     # SKU "1" = the item in every planning pick
    pol, info, files = RS.load_racksmart(cfg, n)
    picked = {TR.A, TR.B, TR.HOT, TR.QUIET}
    (label,) = pol.inp.tours_by_period
    lc = pol.nb.location_clusters(pol.inp, [label], pol.inp.tours_by_period[label])
    assert {pol.faces[i] for i in lc} == picked and pol.plan["faces_with_cluster"] == len(picked)
    assert set(pol.inp.extra["sku"]["month"]) == {"2026-08-21"} and pol.nb.CLUSTERS == cfg["racksmart"]["clusters_path"]
    assert pol.planning_inputs["clusters"]["face_items_end"] == "2026-09-21 00:00:00"
    assert pol.planning_inputs["clusters"]["end"] == "2026-09-21 00:00:00" and str(cfg["racksmart"]["clusters_path"]) in files
    E.check_planning_inputs(pol, pd.Timestamp("2026-09-21"), cfg["lookahead_exceptions"])


def test_clusters_cut_after_the_run_start_are_refused(tmp_path):
    cfg, n = _planning(tmp_path)
    cfg["racksmart"]["clusters_path"] = str(_clusters(tmp_path, cutoff="2026-10-15"))
    pol, _, _ = RS.load_racksmart(cfg, n)
    with pytest.raises(ValueError, match=r"clusters \(up to 2026-10-15"):
        E.check_planning_inputs(pol, pd.Timestamp("2026-09-21"), cfg["lookahead_exceptions"])


def test_snapshot_mode_needs_its_date(tmp_path):
    cfg, n = _planning(tmp_path)
    cfg["racksmart"]["inventory_as_of"] = None
    with pytest.raises(ValueError, match="inventory_as_of"):
        RS.load_racksmart(cfg, n)
    cfg["racksmart"]["move_cases"] = "guess"
    with pytest.raises(ValueError, match="move_cases"):
        RS.load_racksmart(cfg, n)


# --------------------------------------------------------------------------
# the engine's check
# --------------------------------------------------------------------------

def _pol(**inputs):
    p = StaticPolicy()
    p.planning_inputs = inputs
    return p


def test_engine_check_refuses_anything_past_the_start_that_is_not_named():
    start = pd.Timestamp("2026-09-21")
    ok = _pol(a={"end": "2026-09-21"}, s={"end": None, "structural": True})
    assert E.check_planning_inputs(ok, start)["lookahead_exceptions_used"] == {}
    with pytest.raises(ValueError, match=r"b \(up to 2026-09-22"):
        E.check_planning_inputs(_pol(a={"end": "2026-09-21"}, b={"end": "2026-09-22"}), start, {"a": "x"})
    rec = E.check_planning_inputs(_pol(b={"end": "2026-09-22"}), start, {"b": "why"})
    assert rec["lookahead_exceptions_used"] == {"b": "why"} and rec["lookahead_exceptions_allowed"] == {"b": "why"}
    with pytest.raises(ValueError, match="no end date"):
        E.check_planning_inputs(_pol(c={"end": None}), start)
    assert E.check_planning_inputs(StaticPolicy(), start)["inputs"] == {}


# --------------------------------------------------------------------------
# run folders and the comparison
# --------------------------------------------------------------------------

def test_a_pallet_fill_run_gets_its_own_folder():
    cfg = {"window": {"start": "2026-09-21", "end": "2026-10-15"}, "racksmart": {"move_cases": "pallet_fill"}}
    assert E.run_dir_name(cfg, "racksmart", "network_feet") == "sim_racksmart_network_feet_20260921_20261015_pallet_fill"
    assert E.run_dir_name(cfg, "static", "network_feet") == "sim_static_network_feet_20260921_20261015"
    cfg["racksmart"]["move_cases"] = "inventory_snapshot"
    assert E.run_dir_name(cfg, "racksmart", "network_feet") == "sim_racksmart_network_feet_20260921_20261015"


def _manifest(d, policy, net_usd, busy, swaps=3, faces=900, basis="inventory_snapshot", used=("move_cases",)):
    d.mkdir(parents=True)
    m = {"policy": policy, "git_sha": "abc1234def", "git_dirty": False, "moves": {"cost": 10.0 * swaps},
         "descriptive": {"pick_feet": 1000.0 - swaps}, "not_findings": {"busy_hours": busy},
         "planning_inputs": {"lookahead_exceptions_used": {u: "x" for u in used}}}
    if policy != "static":
        m["notebook_plan"] = {"plan": {"swaps": swaps}, "inputs": {"faces": faces, "faces_with_cluster": 5},
                              "move_cases": {"basis": basis},
                              "workbook_replay": {"walking_cut_pct": 1.0, "gross_usd": net_usd + 1, "net_usd": net_usd}}
    (d / "manifest.json").write_text(json.dumps(m))


def test_compare_reports_each_policy_and_whether_the_ranking_changed(tmp_path):
    tag = "20260921_20261015"
    a, b = tmp_path / "a", tmp_path / "b"
    _manifest(a / f"sim_static_network_feet_{tag}", "static", 0, 100.0, swaps=0)
    _manifest(a / f"sim_racksmart_network_feet_{tag}", "racksmart", 500.0, 90.0)
    _manifest(a / f"sim_abc_cost_test_network_feet_{tag}", "abc_cost_test", 300.0, 95.0)
    _manifest(a / "sim_racksmart_network_feet_20261005_20261011", "racksmart", -1.0, 1.0)      # other window
    _manifest(a / f"sim_racksmart_network_feet_{tag}_pallet_fill", "racksmart", 100.0, 99.0, basis="pallet_fill", used=())
    _manifest(a / f"sim_abc_cost_test_network_feet_{tag}_pallet_fill", "abc_cost_test", 200.0, 97.0, basis="pallet_fill", used=())
    snap, _ = C.load_set(a, "2026-09-21", "2026-10-15")
    pf, paths = C.load_set(a, "2026-09-21", "2026-10-15", "pallet_fill")
    assert snap.set_index("policy").at["racksmart", "notebook_net_usd"] == 500.0
    assert pf.set_index("policy").at["racksmart", "move_cases"] == "pallet_fill"
    assert pf.set_index("policy").at["static", "sim_busy_hours"] == 100.0      # static borrowed from the main run
    assert pf.set_index("policy").at["abc_cost_test", "sim_busy_hours_saved_vs_static"] == 3.0
    table, rank = C.compare(snap, pf, "snapshot", "pallet_fill")
    assert rank["notebook_net"] == {"snapshot": ["racksmart", "abc_cost_test"],
                                    "pallet_fill": ["abc_cost_test", "racksmart"], "changed": True}
    assert rank["sim_busy"]["changed"] is True
    row = table[(table.policy == "racksmart") & (table.metric == "notebook_net_usd")].iloc[0]
    assert row["change"] == -400.0
    _, rank_same = C.compare(snap, snap, "x", "y")
    assert not rank_same["notebook_net"]["changed"] and not rank_same["sim_busy"]["changed"]
    _manifest(b / f"sim_racksmart_network_feet_{tag}", "racksmart", 1.0, 1.0)
    with pytest.raises(ValueError, match="no static run"):
        C.load_set(b, "2026-09-21", "2026-10-15")


def test_compare_reads_runs_from_before_the_fix(tmp_path):
    """Old manifests have no planning_inputs and no move_cases: shown as such, not guessed."""
    d = tmp_path / "sim_racksmart_network_feet_20260921_20261015"
    _manifest(d, "racksmart", 5.0, 1.0)
    m = json.loads((d / "manifest.json").read_text())
    del m["planning_inputs"], m["notebook_plan"]["move_cases"]
    (d / "manifest.json").write_text(json.dumps(m))
    _manifest(tmp_path / "sim_static_network_feet_20260921_20261015", "static", 0, 2.0, swaps=0)
    t, _ = C.load_set(tmp_path, "2026-09-21", "2026-10-15")
    r = t.set_index("policy").loc["racksmart"]
    assert r["move_cases"] == "inventory_snapshot" and r["lookahead_exceptions_used"] == "not recorded"


# --------------------------------------------------------------------------
# affinity: the cutoff is recorded in its outputs
# --------------------------------------------------------------------------

def test_graph_json_and_meta_record_the_cutoff(tmp_path):
    import networkx as nx
    from graph_export import graph_json, read_meta, write_meta
    H = nx.Graph()
    H.add_edge("1", "2", weight=0.5, orders=3)
    summary = pd.DataFrame({"skus": [2], "pick_lines": [9], "top_sku": ["x"], "name": ["c"]})
    g = graph_json(H, [{"1", "2"}], {"1": (0, 0), "2": (1, 1)}, summary, 0.1, 1.0, 42, cutoff="2026-09-21")
    assert g["data"]["cutoff_exclusive"] == "2026-09-21"
    write_meta(tmp_path / "m.json", "2026-09-21", made_by="t")
    assert read_meta(tmp_path / "m.json") == {"cutoff_exclusive": "2026-09-21", "made_by": "t"}
    with pytest.raises(SystemExit, match="build_matrix"):
        read_meta(tmp_path / "missing.json")
