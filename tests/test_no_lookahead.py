"""No-lookahead guarantee: a policy may only use orders that have already been
released (and planning inputs that end before the run). Synthetic data only.

1. Guard self-test: a policy that reads a not-yet-released order fails.
2. Prefix invariance (engine): everything decided before time T is identical
   whether or not more orders arrive after T -- including a 'canary' order
   built to change any decision that could see it.
3. Planning invariance (notebook methods): the plan made before the run is
   identical with and without a canary pick on the run's first day in the
   planning file, at a face and item the planning month never touched. Planning
   takes no run-window events at all (load_racksmart has no such argument), and
   every planning input is dated and checked by the engine.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import pytest
import yaml

import test_racksmart as TR
import test_sim as TS
from reslot.sim import engine as E
from reslot.sim.orders import FutureAccessError
from reslot.sim.policy import StaticPolicy, make_policy
from reslot.travel.artifacts import TravelNetwork

net = TS.net                       # shared fixture
T_CUT = 3 * 3600.0                 # orders released before this are 'the past'


class PeekAhead(StaticPolicy):
    """Cheats: on each release, reads the order released after it."""
    name = "peek_ahead"

    def on_arrival(self, order, view):
        view.get(order.tour_id + 1)


class UsesThePast(StaticPolicy):
    """Honest: reads every order released so far."""
    name = "uses_the_past"

    def __init__(self):
        self.seen = []

    def on_arrival(self, order, view):
        self.seen.append([o.tour_id for o in view])


def tours():
    """Six orders through the morning, mostly at the same faces."""
    locs = [("FF001A", "FF003A"), ("FF003A", "GG001A"), ("FF001A", "GG001A")]
    return [TS.tour(*locs[i % 3], tid=i, arrival=1800.0 * i) for i in range(6)]


def canary(tid):
    """Released after T_CUT, with a pick far from everything else and a huge quantity."""
    return TS.tour("GG001A", "FF001A", "GG001A", qtys=[99, 99, 99], tid=tid, arrival=T_CUT + 60)


def run(orders, policy, network):
    return E.simulate(orders, 1, policy, network, TS.Synthetic(), TS.SCOPE, TS.ENDS)


# --------------------------------------------------------------------------
# 1. the guard
# --------------------------------------------------------------------------

def test_a_policy_that_reads_a_future_order_fails(net):
    with pytest.raises(FutureAccessError, match="has not been released"):
        run(tours(), PeekAhead(), net)


def test_an_honest_policy_sees_exactly_the_released_orders(net):
    p = UsesThePast()
    run(tours(), p, net)
    assert p.seen == [list(range(k + 1)) for k in range(6)]


# --------------------------------------------------------------------------
# 2. prefix invariance, for every policy the engine can be handed directly
# --------------------------------------------------------------------------

@pytest.mark.parametrize("policy", [make_policy("static"), UsesThePast()], ids=["static", "uses_the_past"])
def test_decisions_before_t_do_not_depend_on_orders_after_t(net, policy):
    full = tours()
    past = [t for t in full if t.arrival_s < T_CUT]
    a = run(full, policy, net)
    b = run(past + [canary(99)], type(policy)(), net)
    k = a.tours["arrival_s"] < T_CUT
    pd.testing.assert_frame_equal(a.tours[k].reset_index(drop=True),
                                  b.tours[b.tours["arrival_s"] < T_CUT].reset_index(drop=True))
    early = set(a.tours.loc[k, "tour_id"])
    pd.testing.assert_frame_equal(a.legs[a.legs["tour_id"].isin(early)].reset_index(drop=True),
                                  b.legs[b.legs["tour_id"].isin(early)].reset_index(drop=True))


# --------------------------------------------------------------------------
# 3. the notebook methods' plan must not depend on the run window
# --------------------------------------------------------------------------

def _planning_cfg(tmp_path):
    D, xw = TR.network()
    (tmp_path / "travel").mkdir()
    D.to_csv(tmp_path / "travel" / "stop_distance_ft.csv")
    xw.to_csv(tmp_path / "travel" / "location_stop_crosswalk.csv", index=False)
    TR._pick_rows(TR.PLAN, "2026-08-24").assign(CASES_PER_PALLET=60).to_csv(tmp_path / "plan.csv", index=False)
    TR.inventory().to_csv(tmp_path / "inv.csv", index=False)
    cfg = yaml.safe_load((TS.REPO / "configs" / "sim.yaml").read_text())
    cfg["trip_ends"].update(start_stop="D", end_stop="D")
    cfg["racksmart"].update(notebook=str(TR.NOTEBOOK), planning_picks=[str(tmp_path / "plan.csv")],
                            inventory_path=str(tmp_path / "inv.csv"), clusters_path=None)
    return cfg, TravelNetwork(D, xw)


def _canary_rows(path, day="2026-09-21 06:00"):
    """Append picks on the run's first day (Sep 21, the run start) to a planning
    file: a new face (FF005A), a new item, a huge pallet size -- anything that read
    them would change faces, items, cases or tours."""
    df = pd.read_csv(path, dtype=str)
    extra = df.head(3).copy()
    extra["NO"] = [str(10**6 + k) for k in range(3)]
    extra["CREATE_DT"] = pd.Timestamp(day).strftime("%m/%d/%Y %I:%M:%S %p")
    extra["FROM_LOCATION"] = ["FF005A", "FF009A", "FF005A"]
    extra["FROM_ITEM"] = "99999999"
    extra["CASES_PER_PALLET"] = "9999"
    extra["ORD_NO"] = "777"
    pd.concat([df, extra]).to_csv(path, index=False)


def test_planning_takes_no_run_window_events():
    import inspect
    from reslot.sim.racksmart import load_racksmart, prepare_planning, structural_faces
    for f in (load_racksmart, prepare_planning, structural_faces):
        assert not any("event" in name for name in inspect.signature(f).parameters), f.__name__


@pytest.mark.parametrize("move_cases", ["inventory_snapshot", "pallet_fill"])
@pytest.mark.parametrize("method", ["racksmart", "abc_cost_test", "family_grouping"])
def test_the_plan_does_not_depend_on_the_run_window(tmp_path, method, move_cases):
    from reslot.sim.racksmart import load_racksmart
    cfg, n = _planning_cfg(tmp_path)
    cfg["racksmart"]["move_cases"] = move_cases
    p0, i0, _ = load_racksmart(cfg, n, method=method)
    _canary_rows(tmp_path / "plan.csv")
    p1, i1, _ = load_racksmart(cfg, n, method=method)
    assert p0.inp.extra["faces"] == p1.inp.extra["faces"]                       # what the planner could consider
    assert list(p0.inp.cases) == list(p1.inp.cases)
    assert p0.planning_inputs == p1.planning_inputs
    assert i0["planning_workload"] == i1["planning_workload"]
    assert {(m.from_location, m.to_location) for m in p0.moves()} == {(m.from_location, m.to_location) for m in p1.moves()}
    assert i1["planning_rows"]["read"] == i0["planning_rows"]["read"]           # the canary rows were never read


def test_every_planning_input_is_dated_and_checked(tmp_path):
    from reslot.sim.racksmart import load_racksmart
    cfg, n = _planning_cfg(tmp_path)
    pol, _, _ = load_racksmart(cfg, n, method="racksmart")
    ins = pol.planning_inputs
    assert set(ins) == {"planning_picks", "faces", "face_items", "move_cases"}
    assert ins["faces"]["end"] is None and ins["faces"]["structural"]
    assert ins["planning_picks"]["end"] == ins["face_items"]["end"] == "2026-09-21 00:00:00"
    start = pd.Timestamp(cfg["window"]["start"])
    with pytest.raises(ValueError, match="move_cases"):                         # snapshot is after the run start
        E.check_planning_inputs(pol, start, {})
    rec = E.check_planning_inputs(pol, start, cfg["lookahead_exceptions"])
    assert list(rec["lookahead_exceptions_used"]) == ["move_cases"]
    cfg["racksmart"]["move_cases"] = "pallet_fill"
    pol, _, _ = load_racksmart(cfg, n, method="racksmart")
    assert E.check_planning_inputs(pol, start, cfg["lookahead_exceptions"])["lookahead_exceptions_used"] == {}
