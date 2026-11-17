"""No-lookahead guarantee: a policy may only use orders that have already been
released (and planning inputs that end before the run). Synthetic data only.

1. Guard self-test: a policy that reads a not-yet-released order fails.
2. Prefix invariance (engine): everything decided before time T is identical
   whether or not more orders arrive after T -- including a 'canary' order
   built to change any decision that could see it.
3. Planning invariance (notebook methods): the plan made before the run is
   identical with and without a canary pick inside the run window. This fails
   today -- the candidate face set includes faces picked in the run window
   (racksmart.faces_for) -- and is fixed in the next commit.
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
from reslot.schemas import EVENTS, validate
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
    TR._pick_rows(TR.PLAN, "2026-08-24").to_csv(tmp_path / "plan.csv", index=False)
    TR.inventory().to_csv(tmp_path / "inv.csv", index=False)
    cfg = yaml.safe_load((TS.REPO / "configs" / "sim.yaml").read_text())
    cfg["trip_ends"].update(start_stop="D", end_stop="D")
    cfg["racksmart"].update(notebook=str(TR.NOTEBOOK), planning_picks=[str(tmp_path / "plan.csv")],
                            inventory_path=str(tmp_path / "inv.csv"), clusters_path=None)
    return cfg, TravelNetwork(D, xw)


def _run_window_pick(location):
    """One case pick inside the run window (Oct 6), at a face the planning month never touched."""
    return validate(pd.DataFrame([{"event_id": 1, "ts": pd.Timestamp("2026-10-06 08:00"), "event_class": "pick",
                                   "from_zone": "WZPCK", "uom": "CS", "from_loc": location, "qty": 1.0,
                                   "order_id": "9", "user_id": "U"}]), EVENTS, "events", strict=False)


@pytest.mark.xfail(strict=True, reason="face-set leak: racksmart.faces_for includes faces picked in the run "
                                       "window; fixed in the next commit")
@pytest.mark.parametrize("method", ["racksmart", "abc_cost_test", "family_grouping"])
def test_the_plan_does_not_depend_on_the_run_window(tmp_path, method):
    from reslot.sim.racksmart import load_racksmart
    cfg, n = _planning_cfg(tmp_path)
    none = validate(pd.DataFrame(columns=list(EVENTS)), EVENTS, "events", strict=False)
    p0, i0, _ = load_racksmart(cfg, n, none, method=method)
    p1, i1, _ = load_racksmart(cfg, n, _run_window_pick("FF005A"), method=method)
    assert p0.inp.extra["faces"] == p1.inp.extra["faces"]                       # what the planner could consider
    assert {(m.from_location, m.to_location) for m in p0.moves()} == {(m.from_location, m.to_location) for m in p1.moves()}
