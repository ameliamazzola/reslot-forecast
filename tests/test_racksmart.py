"""Tests for the RackSmart policy adapter (src/reslot/sim/racksmart.py) and
scripts/plan_policies.py. The algorithms are notebooks/RackSmart_final.ipynb,
loaded unchanged; these tests check the adapter around it. Synthetic data only.

Synthetic aisle FF: faces FF000A..FF009A, 10 ft apart; dispatch D is 10 ft in
front of FF000A. Planning tours: 200 x [FF000A, FF001A, FF009A] (FF009A is hot
and far from its partners) and 20 x [FF000A, FF001A, FF002A] (FF002A is a quiet
face next to them). The money test should swap FF009A <-> FF002A:
  per hot tour   200 ft -> 60 ft (saves 140 ft) x 200 tours
  per quiet tour  60 ft -> 200 ft (loses 140 ft) x 20 tours
"""
import json
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest
import yaml

from reslot.schemas import EVENTS, validate
from reslot.sim import engine as E
from reslot.sim import racksmart as RS
from reslot.sim.policy import make_policy
from reslot.sim.workload import Pick, Tour
from reslot.travel.artifacts import TravelNetwork

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "RackSmart_final.ipynb"
LOCS = [f"FF{k:03d}A" for k in range(10)]
STOPS = ["D"] + [f"S{k}" for k in range(10)]
SCOPE = {"event_class": "pick", "zone": "WZPCK", "uom": "CS", "slot_type": "F"}
ENDS = {"start_stop": "D", "end_stop": "D", "dispatch_legs": "untimed", "drop_off_mode": "provisional"}
HOT, QUIET, A, B = "FF009A", "FF002A", "FF000A", "FF001A"


def network():
    pos = np.array([-1] + list(range(10)), float)                     # D one step in front of S0
    D = pd.DataFrame(10 * np.abs(pos[:, None] - pos[None, :]), index=pd.Index(STOPS, name="stop_id"),
                     columns=STOPS)
    xw = pd.DataFrame({"location": LOCS, "aisle": "FF", "zone": "WZPCK", "slot_type": "F", "status": "ok",
                       "stop_id": STOPS[1:], "stop_x_in": "180855.63",
                       "stop_y_in": [str(5000 - 120 * k) for k in range(10)]})
    return D, xw


def inventory():
    return pd.DataFrame({"LOCATION": LOCS[:9], "ITEM": [20000000 + k for k in range(9)],
                         "UOM": "CS", "QTY": [5.0] * 9})            # FF009A missing -> median


def interleave(pattern_counts):
    """Spread the patterns evenly over the sequence, the way real tours mix
    across order numbers. (The notebook's backtest guard plans on the first
    half of the tours in order-number order and checks on the second half, so
    a block of one pattern per half would rightly block every swap.)"""
    total = sum(n for _, n in pattern_counts)
    slots = sorted((k * total / n, i) for i, (_, n) in enumerate(pattern_counts) for k in range(n))
    return [pattern_counts[i][0] for _, i in slots]


def tours_of(pattern_counts, start=pd.Timestamp("2026-08-24 08:00")):
    out = []
    for tid, locs in enumerate(interleave(pattern_counts)):
        ts = start + pd.Timedelta(minutes=tid)
        out.append(Tour(tid, f"{800000 + tid}", f"U{tid % 7}", 0.0,
                        tuple(Pick(j, ts, l, 1.0, None) for j, l in enumerate(locs)), 0))
    return out


PLAN = [([A, B, HOT], 200), ([A, B, QUIET], 20)]


@pytest.fixture(scope="module")
def nb():
    return RS.load_notebook(NOTEBOOK)


@pytest.fixture
def net():
    return TravelNetwork(*network())


@pytest.fixture
def planned(nb, net):
    inp, train, built = RS.build_inputs(nb, LOCS, net, "D", inventory(), None, tours_of(PLAN))
    return RS.plan_policy(nb, inp, train), built


# --------------------------------------------------------------------------
# loading the notebook unchanged
# --------------------------------------------------------------------------

def test_notebook_algorithm_cells_run_unchanged_and_nothing_else(nb):
    assert nb.info["cells_run"] == {"settings": 1, "inputs": 3, "engine": 4, "plus": 5, "final": 9}
    for name in ("load_york", "INP", "run_month", "res", "summary"):          # adapter / run cells
        assert name not in nb.ns, name
    assert nb.info["constants"]["COST_PER_CASE"] == 0.17 and nb.info["constants"]["MPH"] == 1.25
    assert len(nb.info["sha256_16"]) == 16


def _notebook(tmp_path, drop=None, poison=None):
    raw = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = raw["cells"]
    if drop is not None:
        cells = [c for c in cells if not ("".join(c["source"])).count(drop)]
    if poison is not None:
        for c in cells:
            if poison in "".join(c["source"]):
                c["source"] = list(c["source"]) + ["\ndef load_york():\n    pass\n"]
    raw["cells"] = cells
    p = tmp_path / "nb.ipynb"
    p.write_text(json.dumps(raw), encoding="utf-8")
    return p


def test_loader_refuses_a_notebook_missing_an_algorithm_cell(tmp_path):
    with pytest.raises(ValueError, match="expected one 'plus' cell"):
        RS.load_notebook(_notebook(tmp_path, drop="class PlannerPlus"))


def test_loader_refuses_to_run_a_cell_that_also_loads_data(tmp_path):
    with pytest.raises(ValueError, match="refusing to run"):
        RS.load_notebook(_notebook(tmp_path, poison="def racksmart_final"))


def test_notebook_file_is_untouched_by_loading(nb):
    from reslot.utils.manifest import file_hash
    assert file_hash(NOTEBOOK) == nb.info["sha256_16"]


# --------------------------------------------------------------------------
# adapter: load_york's rules on our data
# --------------------------------------------------------------------------

def test_inputs_follow_load_york_rules(nb, net):
    inp, train, built = RS.build_inputs(nb, LOCS, net, "D", inventory(), None, tours_of(PLAN))
    n = len(LOCS)
    assert inp.depot == n and inp.D.shape == (n + 1, n + 1)
    assert inp.D[n, 0] == 10 and inp.D[9, n] == 100 and inp.D[0, 9] == 90      # depot = dispatch stop
    assert inp.cases[9] == 5.0                                                # missing -> median
    assert inp.dollars_per_unit == pytest.approx((45.50 / 3600) / (1.25 * 5280 / 3600))
    assert inp.golden == [0, 1, 2, 3, 4, 5, 6]                                # within 60 ft of the front
    assert len(train) == 220 and built["planning_tours"] == 220


def test_visits_merge_repeats_and_drop_short_tours():
    idx = {l: i for i, l in enumerate(LOCS)}
    tours = tours_of([([A, A, B, HOT], 1), ([A, B], 1), ([B, A, A, B], 1)])
    assert RS.visit_sequences(tours, idx) == [[0, 1, 9], [1, 0, 1]]


def test_faces_at_every_level_are_candidates():
    """Team decision: unlike load_york, upper-level faces are not filtered out."""
    D, xw = network()
    xw["level"] = ["A"] * 8 + ["B", "C"]                    # FF008A / FF009A codes, upper-level rows
    ev = validate(pd.DataFrame({"event_id": [1, 2], "ts": pd.Timestamp("2026-08-24"), "event_class": "pick",
                                "from_zone": "WZPCK", "uom": "CS", "from_loc": ["FF008A", "FF009A"],
                                "qty": 1.0, "order_id": "1", "user_id": "U"}), EVENTS, "events", strict=False)
    assert RS.faces_for(TravelNetwork(D, xw), ev, scope=SCOPE) == ["FF008A", "FF009A"]
    assert any("all levels" in d for d in RS.ADAPTER_DIFFERENCES)


def test_depot_is_the_configured_dispatch_stop_not_a_pd_stop(nb, net):
    with pytest.raises(KeyError):
        RS.build_inputs(nb, LOCS, net, "PD_old", inventory(), None, tours_of(PLAN))


# --------------------------------------------------------------------------
# plan + policy
# --------------------------------------------------------------------------

def test_plan_swaps_the_hot_far_face_with_the_quiet_near_one(planned):
    policy, _ = planned
    assert policy.plan["swaps"] == 1 and not policy.plan["blocked_by_backtest_guard"]
    assert policy.location_for(HOT, 0.0) == QUIET and policy.location_for(QUIET, 0.0) == HOT
    assert policy.location_for(A, 0.0) == A


def test_moves_are_reported_and_priced_with_the_notebook_cost(planned):
    policy, _ = planned
    moves = policy.moves()
    assert {(m.from_location, m.to_location) for m in moves} == {(HOT, QUIET), (QUIET, HOT)}
    assert policy.move_cost(moves) == pytest.approx(0.17 * (5.0 + 5.0))      # once per swap, both faces


def test_route_is_the_notebook_replan_and_keeps_every_pick(planned):
    policy, _ = planned
    i = policy.index
    # observed B, A, HOT, HOT. HOT's product now sits at QUIET's face (S2); the
    # notebook re-inserts it at the first cheapest position, in front of B:
    # D->S2->S1 adds 20 ft, same as between B and A, and the first wins.
    expected = policy.world.replan([i[B], i[A], i[HOT]], policy.moved)
    assert expected == [i[QUIET], i[B], i[A]]
    assert policy.route((B, A, HOT, HOT), 0.0) == [2, 3, 0, 1]               # the visit's picks stay together
    assert policy.route((A, HOT, HOT, B), 0.0) == [0, 1, 2, 3]               # tie: first position = as observed
    assert policy.route((A, B), 0.0) == [0, 1]                                # nothing moved: as observed
    assert make_policy("static").route((A, HOT, B), 0.0) == [0, 1, 2]


def test_engine_runs_the_plan_and_charges_it(planned, net):
    policy, _ = planned

    class Synthetic:
        name = "synthetic"

        def predict(self, leg):
            return 10 + 0.1 * leg.feet
    ts = pd.Timestamp("2026-10-05 08:00")
    t = Tour(0, "900001", "U1", 0.0, (Pick(1, ts, A, 1.0, None), Pick(2, ts, HOT, 1.0, 30.0),
                                      Pick(3, ts, B, 1.0, 40.0)), 0)
    static = E.simulate([t], 6, make_policy("static"), net, Synthetic(), SCOPE, ENDS)
    rs = E.simulate([t], 6, policy, net, Synthetic(), SCOPE, ENDS, move_cost=policy.move_cost)
    assert static.tours["pick_feet"][0] + static.tours["dispatch_feet"][0] == 200
    assert rs.tours["pick_feet"][0] + rs.tours["dispatch_feet"][0] == 60      # the notebook's 60 ft tour
    assert rs.move_cost == pytest.approx(1.70)
    assert rs.legs["obs_s"].isna().all()                                      # reordered/moved: not comparable
    assert static.legs["obs_s"].tolist() == [30.0, 40.0]


def test_workbook_replay_is_the_notebook_scoring(planned):
    policy, _ = planned
    r = RS.workbook_replay(policy, tours_of([([A, B, HOT], 100), ([A, B, QUIET], 10)]))
    assert r["feet_before"] == 100 * 200 + 10 * 60
    assert r["feet_saved"] == 100 * 140 - 10 * 140
    dpu = (45.50 / 3600) / (1.25 * 5280 / 3600)
    assert r["gross_usd"] == pytest.approx(round(12600 * dpu, 2)) and r["move_cost_usd"] == pytest.approx(1.7)


# --------------------------------------------------------------------------
# plan_policies.py end to end
# --------------------------------------------------------------------------

def _pick_rows(pattern_counts, day, transactions=False):
    rows, n = [], 0
    for t, locs in enumerate(interleave(pattern_counts)):
        if True:
            for j, loc in enumerate(locs):
                ts = pd.Timestamp(day) + pd.Timedelta(minutes=5 * (n // 3), seconds=30 * j)
                row = {"NO": 0, "TYPE": "9" if transactions else "009",
                       "CREATE_DT": ts.strftime("%m/%d/%Y %I:%M:%S %p"), "USER_NAME": f"U{t % 5}",
                       "FROM_ITEM": "1", "FROM_LOCATION": loc, "TO_LOCATION": "V-1", "FROM_QTY": 1,
                       "FROM_UOM": "CS", "ORD_NO": f"{(9 if transactions else 8) * 100000 + n // 3}",
                       "CUST_NAME": "X", "FROM_ZONE": "WZPCK", "TO_ZONE": "WZEQP"}
                if transactions:
                    row["TRANS_TYPE"] = "PICK"
                rows.append(row)
                n += 1
    df = pd.DataFrame(rows)
    df["NO"] = range(1, len(df) + 1)
    return df


def test_plan_policies_end_to_end(tmp_path):
    D, xw = network()
    (tmp_path / "travel").mkdir()
    D.to_csv(tmp_path / "travel" / "stop_distance_ft.csv")
    xw.to_csv(tmp_path / "travel" / "location_stop_crosswalk.csv", index=False)
    plan = _pick_rows(PLAN, "2026-08-24")
    half = len(plan) // 2                                    # two parts, overlapping row -> dedup by NO
    plan.iloc[: half + 1].to_csv(tmp_path / "p1.csv", index=False)
    plan.iloc[half:].to_csv(tmp_path / "p2.csv", index=False)
    _pick_rows([([A, B, HOT], 50), ([A, B, QUIET], 5)], "2026-10-06", transactions=True
               ).to_csv(tmp_path / "tx.csv", index=False)
    inventory().to_csv(tmp_path / "inv.csv", index=False)
    cfg = yaml.safe_load((REPO / "configs" / "sim.yaml").read_text())
    cfg["data"]["picks_path"] = str(tmp_path / "tx.csv")
    cfg["travel"]["dir"] = str(tmp_path / "travel")
    cfg["trip_ends"].update(start_stop="D", end_stop="D")
    cfg["paths"]["out"] = str(tmp_path / "runs")
    cfg["window"] = {"start": "2026-10-05", "end": "2026-10-11"}      # the synthetic week
    cfg["racksmart"].update(notebook=str(NOTEBOOK), planning_picks=[str(tmp_path / "p1.csv"), str(tmp_path / "p2.csv")],
                            inventory_path=str(tmp_path / "inv.csv"), clusters_path=None)
    (tmp_path / "sim.yaml").write_text(yaml.safe_dump(cfg))

    old = sys.argv
    sys.argv = ["plan_policies.py", str(tmp_path / "sim.yaml")]
    try:
        runpy.run_path(str(REPO / "scripts" / "plan_policies.py"), run_name="__main__")
    finally:
        sys.argv = old
    run_dir = tmp_path / "runs" / "notebook_plans_20261005_20261011"
    m = json.loads((run_dir / "manifest.json").read_text())
    rows = m["planning"]["planning_rows"]
    assert rows["read_window"] == ["2026-08-20", "2026-09-22"] and rows["read"] == len(plan) + 1
    assert rows["after_dedup_by_event_id"] == len(plan) and sum(rows["per_file"].values()) == len(plan) + 1
    assert m["planning"]["notebook"]["cells_run"]["final"] == 9
    assert m["planning"]["adapter_differences"] == RS.ADAPTER_DIFFERENCES
    assert sorted(m["plans"]) == sorted(RS.METHODS)
    for key, p in m["plans"].items():
        assert p["label"] == RS.METHODS[key]["label"] and any("PY/PZ" in g for g in p["known_gaps"])
        assert p["workbook_replay"]["feet_saved"] == 50 * 140 - 5 * 140      # every method finds the one swap here
        swaps = pd.read_csv(run_dir / f"{key}_swaps.csv")
        assert {swaps.loc[0, "location_a"], swaps.loc[0, "location_b"]} == {HOT, QUIET}
    table = pd.read_csv(run_dir / "comparison.csv")
    assert table["method"].tolist() == ["Do nothing"] + [v["label"] for v in RS.METHODS.values()]
    assert len(m["inputs"]) == 3 + 1 + 2 + 1                 # picks, network x2, notebook, planning x2, inventory


# --------------------------------------------------------------------------
# the comparison methods: wired exactly as the notebook's run_month
# --------------------------------------------------------------------------

def test_every_method_matches_the_notebooks_own_run_month(nb, net):
    """Run the notebook's own run_month cell (test only -- the adapter never
    runs it) on the same Inputs, and check each of our plans gives its row."""
    import dataclasses
    raw = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = next("".join(c["source"]) for c in raw["cells"] if "def run_month" in "".join(c["source"]))
    cell = cell[:cell.index("months = sorted")]              # the def only, not the run below it
    ns = dict(nb.ns)
    exec(cell, ns)
    inp, train, _ = RS.build_inputs(nb, LOCS, net, "D", inventory(), None, tours_of(PLAN))
    test = RS.visit_sequences(tours_of([([A, B, HOT], 60), ([A, B, QUIET], 6)]), {l: i for i, l in enumerate(LOCS)})
    nb_inp = dataclasses.replace(inp, tours_by_period={"2026-08": train, "2026-10": test})
    rows, _ = ns["run_month"](nb_inp, "2026-10", nb.DEMAND_SCALE, nb.COST_MULT)
    theirs = {r["method"]: r for r in rows}

    ours = {"racksmart": RS.plan_policy(nb, inp, train), **RS.plan_comparison(nb, inp, train)}
    tours = tours_of([([A, B, HOT], 60), ([A, B, QUIET], 6)])
    for key, pol in ours.items():
        r, t = RS.workbook_replay(pol, tours), theirs[RS.METHODS[key]["label"]]
        assert pol.plan["swaps"] == t["swaps"], key
        assert r["walking_cut_pct"] == t["walking_cut_pct"], key
        assert round(r["gross_usd"]) == t["gross"] and round(r["move_cost_usd"]) == t["move_cost"], key
        assert round(r["feet_saved"] * pol.world.dpu - r["move_cost_usd"]) == t["net"], key
    assert theirs["Do nothing"]["swaps"] == 0


def test_comparison_methods_record_how_they_were_run(nb, net):
    inp, train, _ = RS.build_inputs(nb, LOCS, net, "D", inventory(), None, tours_of(PLAN))
    plans = RS.plan_comparison(nb, inp, train)
    assert set(plans) == {"classic_abc", "abc_cost_test", "family_grouping"}
    for key, pol in plans.items():
        assert pol.name == key and pol.plan["stable"] is False and pol.plan["backtest_guard"] is False
        assert pol.plan["cost_test"] == (key != "classic_abc") and pol.plan["detour_swaps_n_ref"] == 1
        assert pol.location_for(HOT, 0.0) == QUIET                # the one profitable swap, here
    gaps = RS.method_info(plans["classic_abc"], {})["known_gaps"]
    assert any("no cost test" in g for g in gaps) and any("n_ref" in g for g in gaps)
    assert any("affinity clusters" in g for g in RS.METHODS["family_grouping"]["gaps"])
    with pytest.raises(ValueError, match="unknown notebook method"):
        RS.plan_method(nb, inp, train, "reactive_abc")


def test_planning_survives_a_file_with_nothing_in_the_month(tmp_path, nb, net):
    """The archive 365-day file can hold no rows for the planning month: plan from the other."""
    from reslot.sim.racksmart import prepare_planning
    D, xw = network()
    (tmp_path / "travel").mkdir()
    D.to_csv(tmp_path / "travel" / "stop_distance_ft.csv")
    xw.to_csv(tmp_path / "travel" / "location_stop_crosswalk.csv", index=False)
    _pick_rows(PLAN, "2026-08-24").to_csv(tmp_path / "in_month.csv", index=False)
    old = _pick_rows([([A, B, HOT], 5)], "2026-03-02")
    old["NO"] += 10_000
    old.to_csv(tmp_path / "older.csv", index=False)
    inventory().to_csv(tmp_path / "inv.csv", index=False)
    cfg = yaml.safe_load((REPO / "configs" / "sim.yaml").read_text())
    cfg["trip_ends"].update(start_stop="D", end_stop="D")
    cfg["racksmart"].update(notebook=str(NOTEBOOK), planning_picks=[str(tmp_path / "older.csv"), str(tmp_path / "in_month.csv")],
                            inventory_path=str(tmp_path / "inv.csv"), clusters_path=None)
    ev = validate(pd.DataFrame(columns=list(EVENTS)), EVENTS, "events", strict=False)
    _, _, train, base, _ = prepare_planning(cfg, TravelNetwork(D, xw), ev)
    assert len(train) == 220 and base["planning_rows"]["per_file"][str(tmp_path / "older.csv")] == 0
    cfg["racksmart"]["planning_picks"] = [str(tmp_path / "older.csv")]
    with pytest.raises(ValueError, match="no picks between"):
        prepare_planning(cfg, TravelNetwork(D, xw), ev)
