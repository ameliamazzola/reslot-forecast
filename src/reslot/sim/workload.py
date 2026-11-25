"""
Turn canonical pick EVENTS into the tours the simulator replays.

Scope (v1): case picks only -- event_class pick, zone WZPCK, UOM CS. Pallet
picks are out of scope and only counted.

A tour is one operator working one order (config tour.group_keys, the same
keys as the travel calibration). Its picks are visited in observed scan order
(ts, then event_id -- the WMS transaction number -- for scans in the same
second). EVERY tour is simulated, whatever its size; min_picks belongs to the
leg-model check only.

Window: a tour belongs to the window if its first in-scope pick falls inside
[start, end). It is kept whole, even if it runs past the end; a tour that
started before the window is left out whole. Cutting tours at midnight would
invent short tours and lose legs.

Picks that cannot be placed on the travel network are DROPPED, never guessed
at, and every one is logged with the first reason that applies:

    missing_tour_key            no order_id or user_id
    missing_qty                 no quantity (the leg model needs it)
    not_in_crosswalk            location not in the location -> stop crosswalk
    <crosswalk status>          e.g. check_snap (X/Y bays 157-184), aisle_not_drawn
    not_pick_face               maps to a stop, but not a case-pick face in scope
    stop_not_in_distance_table  crosswalk stop missing from the distance table

A dropped pick stays in its tour's observed sequence, so the observed leg into
the next pick is marked unusable (obs_leg_s = None) instead of being merged
with the one before it: a merged leg would carry two picks' handling time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from ..travel.artifacts import TravelNetwork

SUFFIX_RE = re.compile(r"^\d+[A-Z]+$")
DROPPED_COLUMNS = ["event_id", "ts", "order_id", "user_id", "item_id", "from_loc", "qty", "reason"]


@dataclass(frozen=True)
class Pick:
    event_id: int
    ts: pd.Timestamp
    location: str
    qty: float
    # Observed seconds since the previous pick of this tour, or None when that
    # previous pick was dropped or this is the tour's first pick. The
    # leg-model check compares simulated legs against these, nothing else.
    obs_leg_s: float | None
    # The SKU picked (FROM_ITEM); None when unknown. Counts SKU lines per tour:
    # several scan rows of one SKU in one tour are one line.
    item_id: str | None = None


@dataclass(frozen=True)
class Tour:
    tour_id: int
    order_id: str
    user_id: str
    arrival_s: float          # first kept pick scan, seconds after window start
    picks: tuple              # kept picks, in scan order
    n_dropped: int            # this tour's picks dropped as unplaceable

    @property
    def complete(self) -> bool:
        """No pick dropped: observed first-to-last scan span is comparable."""
        return self.n_dropped == 0

    @property
    def n_lines(self) -> int:
        """SKU lines: distinct SKUs among the kept picks (a pick with no SKU is its own line)."""
        known = {p.item_id for p in self.picks if p.item_id is not None}
        return len(known) + sum(p.item_id is None for p in self.picks)

    @property
    def obs_span_s(self) -> float:
        """Observed first-to-last kept scan, seconds."""
        return (self.picks[-1].ts - self.picks[0].ts).total_seconds()


@dataclass
class Workload:
    tours: list
    dropped: pd.DataFrame
    counts: dict


def drop_reasons(picks: pd.DataFrame, net: TravelNetwork, scope: dict, keys: list) -> pd.Series:
    """First applicable drop reason per pick, None where the pick is placeable."""
    xw = net.crosswalk
    loc = picks["from_loc"].astype(object)
    status = loc.map(xw["status"])
    is_face = ((loc.map(xw["zone"]) == scope["zone"])
               & (loc.map(xw["slot_type"]) == scope["slot_type"]))
    in_table = loc.map(xw["in_distance_table"]).fillna(False).astype(bool)

    reason = pd.Series(None, index=picks.index, dtype=object)

    def mark(mask, value):
        mask = mask.fillna(False).astype(bool) & reason.isna()
        reason[mask] = value[mask] if isinstance(value, pd.Series) else value

    mark(picks[keys].isna().any(axis=1), "missing_tour_key")
    mark(picks["qty"].isna(), "missing_qty")
    mark(status.isna(), "not_in_crosswalk")
    mark(status.notna() & (status != "ok"), status)
    mark(~is_face, "not_pick_face")
    mark(~in_table, "stop_not_in_distance_table")
    return reason


def build_workload(events: pd.DataFrame, net: TravelNetwork, scope: dict, tour_cfg: dict,
                   start: pd.Timestamp, end: pd.Timestamp) -> Workload:
    """events: canonical EVENTS. Window is [start, end): pass end as the day
    after the last day."""
    keys = list(tour_cfg["group_keys"])
    in_window = (events["ts"] >= start) & (events["ts"] < end)
    in_scope = ((events["event_class"] == scope["event_class"])
                & (events["from_zone"] == scope["zone"]) & (events["uom"] == scope["uom"]))
    s = events[in_scope].sort_values(["ts", "event_id"], kind="stable").copy()
    s["reason"] = drop_reasons(s, net, scope, keys)

    keyed = s[keys].notna().all(axis=1)
    first_ts = s[keyed].groupby(keys)["ts"].transform("min")
    tour_in_window = pd.Series(False, index=s.index)
    tour_in_window[keyed] = (first_ts >= start) & (first_ts < end)
    sel = s[tour_in_window | (~keyed & (s["ts"] >= start) & (s["ts"] < end))]

    tours, raw = [], []
    for k, g in sel[keyed.reindex(sel.index)].groupby(keys, sort=True):
        kept = g["reason"].isna().to_numpy()
        elapsed = g["ts"].diff().dt.total_seconds().to_numpy()
        picks = [Pick(int(r.event_id), r.ts, str(r.from_loc), float(r.qty),
                      float(elapsed[i]) if i > 0 and kept[i - 1] else None,
                      None if pd.isna(r.item_id) else str(r.item_id).strip())
                 for i, r in enumerate(g.itertuples(index=False)) if kept[i]]
        if picks:
            raw.append((str(k[0]), str(k[1]), picks, int((~kept).sum())))
    raw.sort(key=lambda t: (t[2][0].ts, t[2][0].event_id))
    for i, (order, user, picks, n_dropped) in enumerate(raw):
        tours.append(Tour(i, order, user, (picks[0].ts - start).total_seconds(), tuple(picks), n_dropped))

    dropped = sel[sel["reason"].notna()][DROPPED_COLUMNS].reset_index(drop=True)
    return Workload(tours, dropped, _counts(events, in_window, in_scope, s, sel, keyed, tours, dropped,
                                           keys, start, end))


def _counts(events, in_window, in_scope, s, sel, keyed, tours, dropped, keys, start, end) -> dict:
    n_sel = len(sel)
    span = s[keyed].groupby(keys)["ts"].agg(["min", "max"])
    out_of_scope = events[in_window & ~in_scope]
    by_user = {}
    for t in tours:
        by_user.setdefault(t.user_id, []).append((t.picks[0].ts, t.picks[-1].ts))
    overlapping = 0
    for spans in by_user.values():
        spans.sort()
        last_end = None
        for a, b in spans:
            if last_end is not None and a < last_end:
                overlapping += 1
            last_end = b if last_end is None else max(last_end, b)
    return {
        "picks_in_window_tours": int(n_sel),
        "picks_kept": int(sum(len(t.picks) for t in tours)),
        "picks_dropped": int(len(dropped)),
        "dropped_share": float(len(dropped) / n_sel) if n_sel else 0.0,
        "dropped_by_reason": {str(k): int(v) for k, v in dropped["reason"].value_counts().items()},
        "tours": len(tours),
        "tours_with_dropped_picks": int(sum(not t.complete for t in tours)),
        "tours_single_pick": int(sum(len(t.picks) == 1 for t in tours)),
        "tours_all_picks_dropped": int(sel[keyed.reindex(sel.index)].groupby(keys)["reason"]
                                       .apply(lambda r: r.notna().all()).sum()) if n_sel else 0,
        "tours_running_past_window_end": int(sum(t.picks[-1].ts >= end for t in tours)),
        # left out whole: began before start, still picking after it
        "tours_started_before_window": int(((span["min"] < start) & (span["max"] >= start)).sum()),
        "tours_overlapping_same_user": overlapping,
        # e.g. 10906764CR / LR: meaning unconfirmed (data-notes); kept, counted
        "orders_with_alpha_suffix": int(sum(bool(SUFFIX_RE.match(t.order_id)) for t in tours)),
        "out_of_scope_in_window": {f"{c}/{z}/{u}": int(n) for (c, z, u), n in
                                   out_of_scope.groupby(["event_class", "from_zone", "uom"], dropna=False)
                                   .size().sort_values(ascending=False).head(10).items()},
    }
