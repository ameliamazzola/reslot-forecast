"""
Where the engine gets its orders from, and what a policy may see of them.

The engine pulls orders from an OrderSource, one at a time, in release order.
Two modes share the same engine:

  replay  ReplaySource -- the transactions export over a date window. Fast,
          seeded, citable (manifest mode: replay).
  live    a source that releases orders as they come: GeneratorSource (the
          team's OrderGenerator, src/reslot/orders.py) or LiveFeedSource (a
          real feed, stub). Runs on SimPy's real-time clock; never citable
          (manifest mode: live).

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
(engine-checked, input by input, through planning_inputs; an input past the run
start must be named in config lookahead_exceptions), (2) the orders that have already
been released, through the ArrivedOrders view, and (3) the order whose tour is
starting. The view raises FutureAccessError for anything not yet released.
tests/test_no_lookahead.py holds every policy to this.
"""

from __future__ import annotations

from typing import Iterator, Protocol

import pandas as pd

from .workload import Pick, Tour, Workload, build_workload

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


GENERATOR_RELEASED_AT = "the generator's Order.minute x 60 s after the run start (orders in one minute share it)"


class GeneratorSource:
    """Orders from the team's OrderGenerator (src/reslot/orders.py), as engine
    orders. Never citable (manifest mode: live).

    Each generated Order (minute, [(sku, cases)]) becomes one Tour: arrival_s =
    released_at = minute x 60, one Pick per SKU line in the order drawn, at the
    SKU's face (sku_face), qty = cases; user_id None -- the operator pool assigns
    the operator (docs/order_feed_format.md). One order = one tour.

    Minutes walked: mode 'default' -- the team generator as written, each day's shift
    (shift_start_hour, generator.shift_minutes long), multiplier 1; mode
    'empirical' -- every minute, multiplier from the hour-of-day and day-of-week
    shares (generator.minute_multiplier); hour_share is one 24-list or
    {"weekday": [...], "weekend": [...]}.

    Orders are generated lazily, one minute at a time, as the engine pulls them,
    so a control change (set_affinity / set_speed) applies to every order
    generated after it. Every change is logged with its sim time and the number
    of orders generated so far; describe() carries the log into the manifest.
    """
    kind = "generator"
    citable = False

    def __init__(self, generator, sku_face: dict, start: pd.Timestamp, days: int, mode: str = "default",
                 hour_share: list | dict | None = None, day_share: list | None = None, shift_start_hour: int = 6,
                 layout: dict | None = None, inputs: list | None = None, keep_orders: bool = True,
                 label: str | None = None):
        if mode not in ("default", "empirical"):
            raise ValueError(f"generator mode must be 'default' or 'empirical', got {mode!r}")
        if mode == "empirical" and (hour_share is None or day_share is None):
            raise ValueError("empirical mode needs hour_share and day_share")
        missing = sorted(set(generator.skus) - set(sku_face))
        if missing:
            raise ValueError(f"{len(missing)} generator SKUs have no face, e.g. {missing[:3]}")
        self.gen, self.sku_face, self.mode = generator, dict(sku_face), mode
        self.label = label or mode
        self.start, self.days = pd.Timestamp(start), int(days)
        self.hour_share, self.day_share, self.shift_start_hour = hour_share, day_share, int(shift_start_hour)
        self.layout = dict(layout or {})
        self.inputs = list(inputs or [])
        self.keep_orders = keep_orders
        self.generated: list = []            # the generator's own Orders, as drawn
        self.controls: list = []             # every live control change
        self.speed = 1.0
        self.affinity_initial = float(generator.affinity)
        self._n_tours = 0
        self._event = 0

    # -- what the engine pulls ---------------------------------------------
    def _minutes(self):
        if self.mode == "empirical":
            yield from range(self.days * 1440)
            return
        first = self.shift_start_hour * 60
        for d in range(self.days):
            for m in range(first, min(first + self.gen.shift_minutes, 1440)):
                yield d * 1440 + m

    def _multiplier(self, minute: int) -> float:
        if self.mode == "default":
            return 1.0
        from .generator import minute_multiplier
        return minute_multiplier(self.gen, self.start + pd.Timedelta(minutes=minute), self.hour_share, self.day_share)

    def _tour(self, o) -> Tour:
        ts = self.start + pd.Timedelta(minutes=o.minute)
        picks = []
        for sku, cases in o.lines:
            self._event += 1
            picks.append(Pick(self._event, ts, self.sku_face[sku], float(cases), None))
        t = Tour(self._n_tours, o.order_id, None, float(o.minute * 60), tuple(picks), 0)
        self._n_tours += 1
        return t

    def orders(self) -> Iterator[Tour]:
        for minute in self._minutes():
            for o in self.gen.arrivals(minute, self._multiplier(minute)):
                if self.keep_orders:
                    self.generated.append(o)
                yield self._tour(o)

    # -- live controls --------------------------------------------------------
    def _log(self, control: str, value: float, previous: float, sim_time: float) -> None:
        self.controls.append({"sim_time_s": float(sim_time), "control": control, "value": float(value),
                              "previous": float(previous), "orders_generated_before": self._n_tours})

    def set_affinity(self, value: float, sim_time: float) -> None:
        prev = self.gen.affinity
        self.gen.set_affinity(value)
        self._log("affinity", value, prev, sim_time)

    def set_speed(self, value: float, sim_time: float) -> None:
        """Real-time speed factor for the live runner (recorded here so one log has every control)."""
        if not value > 0:
            raise ValueError("speed must be > 0")
        prev, self.speed = self.speed, float(value)
        self._log("speed", value, prev, sim_time)

    def describe(self) -> dict:
        return {"type": self.kind, "mode": self.mode, "mode_label": self.label, "citable": False,
                "start": str(self.start),
                "days": self.days, "released_at": GENERATOR_RELEASED_AT,
                "operator": "none on the order; the engine's operator pool assigns one",
                "orders_generated": self._n_tours, "skus": len(self.gen.skus),
                "profile_source": getattr(self.gen.profile, "source", None),
                "affinity_initial": self.affinity_initial, "affinity_now": float(self.gen.affinity),
                "layout": self.layout, "controls": list(self.controls)}
