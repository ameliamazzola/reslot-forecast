"""
Location -> stop crosswalk: which drawn stop a truck works each WMS location from.

The WMS has no coordinates, so this is a rule, not a lookup:

  * AISLE: each aisle code sits on one drawn path, keyed by x (config).
    FF is the rightmost (east) aisle; letters run west.
  * BAY: bay 001 is at the dock (top of the drawing, high y) and numbers run
    toward the chargers. One pallet per bay; odd bays on the east face, even on
    the west, so each 103" rack section -- one stop -- holds 4 bay numbers.
  * CROSS-AISLES are tunnels: bays 79-84 and 185-190 exist only at upper
    levels (D/E), over the upper and middle cross-aisles. Their stop is the
    cross-aisle junction underneath.
  * LEVEL does not change the stop. It is carried for lift time later.

Each segment of an aisle (top third, half section, tunnels, middle and bottom
thirds) is anchored to a drawn y, so numbering errors cannot accumulate across
the aisle. Anchors live in configs/travel.yaml.

Evidence the rule is right: every building-column rack section on the drawing
sits exactly where the WMS skips a bay number (column_check), and no stop gets
more than 4 floor locations. Both are recorded in the manifest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def target_y(bay: int, segments: list[dict], pitch_in: float, bays_per_stop: int):
    """(segment, expected y, node type) for a bay number, or (None, nan, None)."""
    for s in segments:
        if s["first_bay"] <= bay <= s["last_bay"]:
            step = (bay - s["first_bay"]) // bays_per_stop if s.get("stepped", True) else 0
            return s["name"], s["first_y"] - pitch_in * step, s["node"]
    return None, float("nan"), None


def build_crosswalk(slots: pd.DataFrame, stops: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per location in a mapped aisle (every level), plus unmapped
    rows with a status explaining why.

    slots: canonical SLOTS (needs location, aisle); bay/level parsed here.
    stops: output of network.build_graph.
    """
    from ..io.onetrack import parse_location

    rules = cfg["bay_rules"]
    aisle_x = {a: (x if isinstance(x, list) else [x]) for a, x in cfg["aisle_x"].items()}
    odd_side = rules["odd_side"]
    even_side = "west" if odd_side == "east" else "east"

    parsed = slots["location"].map(parse_location).apply(pd.Series)
    df = pd.concat([slots[["location", "zone", "slot_type"]].reset_index(drop=True),
                    parsed.reset_index(drop=True)], axis=1)

    by_aisle = {a: stops[stops["x_in"].apply(lambda v: any(abs(v - x) < 0.5 for x in xs))]
                for a, xs in aisle_x.items()}
    out = []
    for r in df.itertuples(index=False):
        row = {"location": r.location, "aisle": r.aisle, "bay": r.bay, "level": r.level,
               "zone": r.zone, "slot_type": r.slot_type}
        if r.aisle is None or pd.isna(r.bay):
            out.append({**row, "status": "not_a_rack_code"})
            continue
        if r.aisle not in aisle_x:
            out.append({**row, "status": "aisle_not_drawn"})
            continue
        seg, ty, node = target_y(int(r.bay), rules["segments"], rules["pitch_in"],
                                 rules["bays_per_stop"])
        cand = by_aisle[r.aisle]
        cand = cand[cand["node_type"] == node] if node else cand.iloc[0:0]
        if seg is None or cand.empty:
            out.append({**row, "status": "no_rule_or_stop"})
            continue
        s = cand.loc[(cand["y_in"] - ty).abs().idxmin()]
        snap = abs(s["y_in"] - ty)
        out.append({**row, "side": odd_side if r.bay % 2 else even_side, "segment": seg,
                    "stop_id": s["stop_id"], "stop_x_in": round(s["x_in"], 2),
                    "stop_y_in": round(s["y_in"], 2), "node_type": s["node_type"],
                    "snap_in": round(snap, 1),
                    "status": "ok" if snap <= rules["snap_tol_in"] else "check_snap"})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def floor_load(xw: pd.DataFrame, floor_level: str = "A") -> dict:
    """Floor locations per rack stop. More than bays_per_stop means the bay
    rule is off somewhere."""
    f = xw[(xw["status"] == "ok") & (xw["level"] == floor_level) & (xw["node_type"] == "rack")]
    n = f.groupby("stop_id").size()
    return {"max_floor_locations_per_stop": int(n.max()) if len(n) else 0,
            "stops_over_4": n[n > 4].index.tolist()}


def column_check(xw: pd.DataFrame, columns, cfg: dict, radius_in: float = 100.0) -> dict:
    """Do the drawn column sections sit where the WMS skips bay numbers?

    For each aisle, a skipped bay is a number in 1..max_bay absent from the
    master. Every column section drawn beside that aisle (within radius_in of
    its path, on either face) should land on the expected y of a skipped bay.
    Independent evidence: the drawing and the WMS were made by different
    people for different reasons.
    """
    rules = cfg["bay_rules"]
    cols = np.asarray(columns, float).reshape(-1, 2)
    res, matched, total = {}, 0, 0
    for a, xs in cfg["aisle_x"].items():
        xs = xs if isinstance(xs, list) else [xs]
        near = cols[np.min(np.abs(cols[:, [0]] - np.array(xs)[None, :]), axis=1) < radius_in] \
            if len(cols) else cols
        if not len(near):
            continue
        present = set(xw.loc[xw["aisle"] == a, "bay"].dropna().astype(int))
        skipped = [b for b in range(1, rules["max_bay"] + 1) if b not in present]
        ys = [target_y(b, rules["segments"], rules["pitch_in"], rules["bays_per_stop"])[1]
              for b in skipped]
        hit = sum(any(abs(cy - y) < 2 for y in ys) for cy in near[:, 1])
        res[a] = f"{hit}/{len(near)}"
        matched, total = matched + hit, total + len(near)
    return {"column_sections_explained": f"{matched}/{total}", "by_aisle": res}


def pick_coverage(events: pd.DataFrame, xw: pd.DataFrame, zone: str, uom: str) -> dict:
    """Share of in-scope picks whose location maps to a stop. The number that
    says whether the drawn area is the right scope."""
    s = events[(events["event_class"] == "pick") & (events["uom"] == uom)]
    ok = set(xw.loc[xw["status"] == "ok", "location"])
    in_zone = s[s["from_zone"] == zone]
    return {"uom": uom, "picks": int(len(s)), "mapped": int(s["from_loc"].isin(ok).sum()),
            "coverage": float(s["from_loc"].isin(ok).mean()) if len(s) else 0.0,
            "coverage_in_zone": float(in_zone["from_loc"].isin(ok).mean()) if len(in_zone) else 0.0,
            "top_unmapped_aisles": s.loc[~s["from_loc"].isin(ok), "from_loc"]
                                    .str.extract(r"^([A-Z]+)")[0].value_counts().head(8).to_dict()}
