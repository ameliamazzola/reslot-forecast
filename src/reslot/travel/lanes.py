"""
Outbound staging lanes: which drawn lane stops each WMS pallet-lane code uses.

The WMS has one location per dock staging area (PL01 ... PL57, zone WZNDK);
the drawing has one Stop_Lane circle per physical lane (100, in groups of 4).
They don't line up one-to-one, so the mapping is a stated convention, not a
measurement. Agreed rules (Anthony, Oct 19):

  * PL01 is the rightmost (east) lane group; numbers ascend to the left (west).
  * Lanes per PL come from WMS capacity: 44 pallets -> 2 lanes, 66 or 88 -> 3.
    That asks for 131 lanes and only 100 are drawn, so...
  * ...one physical lane may serve two neighbouring PL codes.
  * PL..A codes (PL08A, 13A, ...) are inbound receiving staging and PL60 has
    no lane: both are left out here.

Within those rules the placement is chosen to keep each PL's lanes inside one
lane group (a PL split across the 13-ft gap between groups is the thing that
would distort travel). Neighbours share at most one lane. Five PLs still have
to straddle a gap; they are listed in the build output.

This is PROVISIONAL until the site confirms which lanes sit in front of which
door. Every number that uses a lane stop inherits that caveat.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


def lane_stops(stops: pd.DataFrame, group_gap_in: float) -> pd.DataFrame:
    """Lane stops numbered 1..N from the right, with their lane group."""
    ln = stops[stops["node_type"] == "lane"].sort_values("x_in", ascending=False).reset_index(drop=True)
    ln["lane_no"] = np.arange(1, len(ln) + 1)
    gaps = -np.diff(ln["x_in"].to_numpy()) > group_gap_in
    ln["lane_group"] = np.concatenate([[1], 1 + np.cumsum(gaps)]).astype(int)
    return ln[["lane_no", "lane_group", "stop_id", "x_in", "y_in"]]


def pl_lane_counts(slots: pd.DataFrame, pattern: str, three_lane_min_capacity: float) -> pd.DataFrame:
    """PL codes in WMS order with the number of lanes each one gets."""
    rx = re.compile(pattern)
    pl = slots[slots["location"].astype(str).map(lambda s: bool(rx.fullmatch(s)))]
    pl = pl[["location", "capacity"]].sort_values("location").reset_index(drop=True)
    pl["n_lanes"] = np.where(pl["capacity"] >= three_lane_min_capacity, 3, 2)
    return pl


def assign_pl_lanes(lanes: pd.DataFrame, pl: pd.DataFrame) -> pd.DataFrame:
    """Place each PL on a contiguous run of lanes, in order, right to left.

    Dynamic programme over start lanes: PL i+1 starts right after PL i ends
    or shares PL i's last lane (overlap 0 or 1). Every lane is covered, the
    first PL starts at lane 1, the last ends at lane N. Among placements that
    satisfy that, pick the one with the fewest PLs split across a lane-group
    gap. Ties go to the earliest feasible start, so the result is deterministic.
    """
    N, n = len(lanes), pl["n_lanes"].to_numpy()
    K = len(n)
    group = dict(zip(lanes["lane_no"], lanes["lane_group"]))
    INF = 10**9

    def split(s: int, k: int) -> int:
        return int(group[s] != group[s + k - 1])

    best = np.full((K, N + 2), INF)
    prev = np.zeros((K, N + 2), dtype=int)
    best[0][1] = split(1, n[0])
    for i in range(1, K):
        for s in range(1, N - n[i] + 2):
            for overlap in (0, 1):
                ps = s - n[i - 1] + overlap
                if ps >= 1 and best[i - 1][ps] < INF:
                    c = best[i - 1][ps] + split(s, n[i])
                    if c < best[i][s]:
                        best[i][s], prev[i][s] = c, ps
    end_start = N - n[-1] + 1
    if best[K - 1][end_start] >= INF:
        raise ValueError(f"{K} PL codes needing {n.sum()} lanes cannot cover {N} lanes "
                         "with at most one shared lane between neighbours")
    starts, s = [0] * K, end_start
    for i in range(K - 1, -1, -1):
        starts[i], s = s, prev[i][s]

    by_no = lanes.set_index("lane_no")
    rows = []
    for (loc, cap, k), s in zip(pl[["location", "capacity", "n_lanes"]].itertuples(index=False), starts):
        nos = list(range(s, s + k))
        mid = nos[(k - 1) // 2] if k == 3 else nos[0]
        rows.append({"location": loc, "capacity": cap, "n_lanes": k,
                     "first_lane": s, "last_lane": s + k - 1,
                     "lane_groups": sorted({int(by_no.loc[x, "lane_group"]) for x in nos}),
                     "lane_stop_ids": [by_no.loc[x, "stop_id"] for x in nos],
                     "stop_id": by_no.loc[mid, "stop_id"]})
    out = pd.DataFrame(rows)
    out["split_across_gap"] = out["lane_groups"].map(len) > 1
    return out


def lane_labels(lanes: pd.DataFrame, assignment: pd.DataFrame) -> pd.DataFrame:
    """One row per physical lane with every PL code it serves."""
    lab = {no: [] for no in lanes["lane_no"]}
    for r in assignment.itertuples(index=False):
        for no in range(r.first_lane, r.last_lane + 1):
            lab[no].append(r.location)
    out = lanes.copy()
    out["pl_codes"] = out["lane_no"].map(lambda x: "/".join(lab[x]))
    return out


def apply_to_crosswalk(xw: pd.DataFrame, assignment: pd.DataFrame) -> pd.DataFrame:
    """Give PL locations a stop: the middle lane of a 3-lane PL, the right-hand
    lane of a 2-lane PL. All of a PL's lanes are kept in lane_stop_ids."""
    m = assignment.set_index("location")
    hit = xw["location"].isin(m.index)
    xw = xw.copy()
    xw.loc[hit, "stop_id"] = xw.loc[hit, "location"].map(m["stop_id"])
    xw.loc[hit, "node_type"] = "lane"
    xw.loc[hit, "segment"] = "staging_lane"
    xw.loc[hit, "status"] = "ok_provisional"
    return xw
