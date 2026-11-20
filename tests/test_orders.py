"""Tests for order sources and the pull-based engine (src/reslot/sim/orders.py,
engine.simulate). Synthetic data only; fixtures shared with test_sim."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import pytest
import simpy

import test_sim as TS
from reslot.sim import engine as E
from reslot.sim.orders import (REPLAY_RELEASED_AT, ArrivedOrders, FutureAccessError, LiveFeedSource,
                               ReplaySource)
from reslot.sim.policy import make_policy
from reslot.sim.workload import build_workload

net = TS.net                       # shared fixture
EV = [("O1", "U1", "2026-10-05 08:00:00", "FF001A", 1), ("O1", "U1", "2026-10-05 08:01:00", "FF003A", 1),
      ("O2", "U2", "2026-10-05 08:00:00", "GG001A", 2), ("O2", "U2", "2026-10-05 08:02:00", "FF001A", 1),
      ("O3", "U3", "2026-10-06 09:00:00", "FF003A", 1), ("O3", "U3", "2026-10-06 09:03:00", "GG001A", 1)]


def source(net):
    return ReplaySource(TS.events(EV), net, TS.SCOPE, TS.TOUR, TS.START, TS.END, picks_path="tx.csv")


def test_replay_source_is_todays_workload_in_release_order(net):
    src, work = source(net), build_workload(TS.events(EV), net, TS.SCOPE, TS.TOUR, TS.START, TS.END)
    assert [t.tour_id for t in src.orders()] == [t.tour_id for t in work.tours]
    assert [t.arrival_s for t in src.orders()] == sorted(t.arrival_s for t in work.tours)
    d = src.describe()
    assert d["type"] == "replay" and d["released_at"] == REPLAY_RELEASED_AT and "first kept pick scan" in d["released_at"]
    assert d["window"] == ["2026-10-05", "2026-10-11"] and src.inputs == ["tx.csv"] and src.citable


def test_engine_gives_the_same_result_from_a_source_or_a_list(net):
    src = source(net)
    a = E.simulate(src, 1, make_policy("static"), net, TS.Synthetic(), TS.SCOPE, TS.ENDS)
    b = E.simulate(list(src.tours), 1, make_policy("static"), net, TS.Synthetic(), TS.SCOPE, TS.ENDS)
    pd.testing.assert_frame_equal(a.tours, b.tours)
    pd.testing.assert_frame_equal(a.legs, b.legs)


def test_a_source_that_goes_back_in_time_is_refused(net):
    class Backwards:
        def orders(self):
            yield TS.tour("FF001A", "FF003A", tid=0, arrival=100.0)
            yield TS.tour("FF001A", "FF003A", tid=1, arrival=50.0)
    with pytest.raises(ValueError, match="before the previous order"):
        E.simulate(Backwards(), 1, make_policy("static"), net, TS.Synthetic(), TS.SCOPE, TS.ENDS)


def test_observer_sees_every_release_start_and_finish_in_time_order(net):
    seen = []
    E.simulate(source(net), 1, make_policy("static"), net, TS.Synthetic(), TS.SCOPE, TS.ENDS,
               observer=lambda ev, p: seen.append((ev, p["t"], p["tour_id"])))
    for kind in ("release", "start", "done"):
        assert sorted(x[2] for x in seen if x[0] == kind) == [0, 1, 2]
    assert [x[1] for x in seen] == sorted(x[1] for x in seen)


def test_the_environment_can_be_supplied(net):
    env = simpy.Environment(initial_time=0)
    r = E.simulate(source(net), 2, make_policy("static"), net, TS.Synthetic(), TS.SCOPE, TS.ENDS, env=env)
    assert env.now == r.tours["end_s"].max()


def test_arrived_orders_only_holds_released_orders():
    env = simpy.Environment()
    view = ArrivedOrders(env)
    early, late = TS.tour("FF001A", tid=0, arrival=0.0), TS.tour("FF001A", tid=1, arrival=60.0)
    view._release(early)
    assert len(view) == 1 and view.get(0) is early and list(view) == [early] and view.recent(5) == [early]
    with pytest.raises(FutureAccessError, match="not been released"):
        view.get(1)
    with pytest.raises(FutureAccessError):
        view._release(late)                    # the engine may not release an order early either


def test_run_records_mode_and_source_and_refuses_a_plan_that_saw_the_run(net):
    out = E.run(TS.cfg(), TS.events(EV), net, TS.Synthetic())
    assert out.extra["mode"] == "replay" and out.extra["source"]["type"] == "replay"
    pol = make_policy("static")
    pol.planning_inputs_end = pd.Timestamp("2026-10-06")          # after the run start (Oct 5)
    with pytest.raises(ValueError, match="after the run start"):
        E.run(TS.cfg(), TS.events(EV), net, TS.Synthetic(), policy=pol)


def test_live_feed_source_is_a_documented_stub():
    with pytest.raises(NotImplementedError, match="order_feed_format.md"):
        LiveFeedSource()
