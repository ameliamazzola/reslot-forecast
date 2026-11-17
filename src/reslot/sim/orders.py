"""
Where the engine gets its orders from, and what a policy may see of them.

The engine pulls orders from an OrderSource, one at a time, in release order.
Two modes share the same engine:

  replay  ReplaySource -- the transactions export over a date window. Fast,
          seeded, citable (manifest mode: replay).
  live    a source that releases orders as they come (GeneratorSource, later;
          LiveFeedSource for a real feed). Runs on SimPy's real-time clock;
          never citable (manifest mode: live).

The engine's order representation is the workload's Tour (one released work
unit) with its Picks (the order lines, in pick order). The message a real feed
would send, and how it maps onto a Tour, is docs/order_feed_format.md.

released_at in replay
---------------------
The transactions export has no order-release time. ReplaySource releases each
tour at its FIRST KEPT PICK SCAN: a proxy for when the WMS released the work,
later than the true release by however long the operator took to reach the
first pick. Every replay manifest records this (source.released_at).

No lookahead
------------
A policy may only use: (1) planning inputs that end at or before the run start
(engine-checked through planning_inputs_end), (2) the orders that have already
been released, through the ArrivedOrders view, and (3) the order whose tour is
starting. The view raises FutureAccessError for anything not yet released.
tests/test_no_lookahead.py holds every policy to this.
"""

from __future__ import annotations

from typing import Iterator, Protocol

import pandas as pd

from .workload import Tour, Workload, build_workload

REPLAY_RELEASED_AT = ("first kept pick scan of each tour (order x operator): a proxy for the WMS "
                      "release time, later than the true release by the drive to the first pick")


class FutureAccessError(RuntimeError):
    """A policy tried to read an order that has not been released yet."""


class OrderSource(Protocol):
    """Yields orders (workload.Tour) in non-decreasing release time.

    kind:      'replay' | 'generator' | 'live_feed'
    citable:   only a replay of real data can be cited
    orders():  the orders, in release order (arrival_s, then tour_id)
    describe(): the manifest's source block
    inputs:    files to hash into the manifest
    """
    kind: str
    citable: bool
    inputs: list

    def orders(self) -> Iterator[Tour]: ...
    def describe(self) -> dict: ...


class ReplaySource:
    """The transactions export over a window: today's workload, released in order.

    Built on build_workload, so the tours, their order and the dropped-pick log
    are exactly what the engine replayed before sources existed.
    """
    kind = "replay"
    citable = True

    def __init__(self, events: pd.DataFrame, net, scope: dict, tour_cfg: dict,
                 start: pd.Timestamp, end: pd.Timestamp, picks_path: str | None = None):
        self.start, self.end = start, end
        self.work: Workload = build_workload(events, net, scope, tour_cfg, start, end)
        self.inputs = [picks_path] if picks_path else []
        self._picks_path = picks_path

    @property
    def tours(self) -> list:
        return self.work.tours

    @property
    def dropped(self) -> pd.DataFrame:
        return self.work.dropped

    @property
    def counts(self) -> dict:
        return self.work.counts

    def orders(self) -> Iterator[Tour]:
        yield from sorted(self.work.tours, key=lambda t: (t.arrival_s, t.tour_id))

    def describe(self) -> dict:
        return {"type": self.kind, "picks_path": self._picks_path,
                "window": [str(self.start.date()), str((self.end - pd.Timedelta(days=1)).date())],
                "orders": len(self.work.tours), "released_at": REPLAY_RELEASED_AT}


class LiveFeedSource:
    """Stub for a real order feed. Not implemented.

    It would read order messages in the docs/order_feed_format.md format (JSON
    lines, one 'order.release' message per line, from a socket or a tailed file),
    validate and place them exactly as ReplaySource places picks (same drop
    reasons), and yield one Tour per message as it arrives, with user_id None
    (the operator pool assigns the operator). It must never yield an order
    before its released_at, and must reject one released earlier than the last.
    """
    kind = "live_feed"
    citable = False
    inputs: list = []

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("LiveFeedSource is a documented stub: see docs/order_feed_format.md")


class ArrivedOrders:
    """What a policy may see of the orders: those already released, nothing else.

    The engine adds each order at its release time. Reading an order that is
    not here -- not yet released, or not an order at all -- raises
    FutureAccessError, so a policy that tries to look ahead fails loudly.
    """

    def __init__(self, env):
        self._env = env
        self._orders: list = []
        self._by_id: dict = {}

    def _release(self, order) -> None:          # engine only
        if order.arrival_s > self._env.now + 1e-9:
            raise FutureAccessError(f"order {order.tour_id} released at {order.arrival_s} before t={self._env.now}")
        self._orders.append(order)
        self._by_id[order.tour_id] = order

    def __len__(self) -> int:
        return len(self._orders)

    def __iter__(self):
        return iter(list(self._orders))

    def get(self, tour_id: int):
        try:
            return self._by_id[tour_id]
        except KeyError:
            raise FutureAccessError(f"order {tour_id} has not been released at t={self._env.now}") from None

    def recent(self, n: int) -> list:
        return list(self._orders[-n:])
