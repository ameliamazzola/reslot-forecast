"""Planning picks from several WMS exports, and the check that they agree.

The planning window can span exports: the 365-day pick reports (TYPE 009, no
TRANS_TYPE) and the transactions exports (TRANS_TYPE, TYPE 9). load_wms_picks
reads both into canonical EVENTS; this module

  read_picks     reads each file over one window (each entry a path, or
                 {path, ts_format}), records what each contributed
  dedup          drops rows that are the same WMS transaction seen in two
                 exports (DEDUP_KEY: transaction number + what it did)
  source_check   for each configured overlap, builds the case-pick legs each
                 side sees and refuses to plan unless they agree in count and
                 in time within the configured tolerance -- and unless the
                 deduplicated union gives the same legs as either side alone

Legs here are raw pick-to-pick legs: consecutive in-scope picks of one
(order, operator), ordered by (ts, event_id), both picks inside the compared
window. No crosswalk, no filters -- the check is about the exports, not the
network. Both sides are cut to the same window before legs are built, so a
tour truncated by the window edge is truncated the same way on both.

The compared window is the configured overlap clipped to the span both sides
actually hold (an export that starts at 12:03 rather than noon); the check
records both and refuses if the clipped span is under tolerance.min_cover of
the configured one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..io.wms_picks import load_wms_picks

DEDUP_KEY = ["event_id", "event_class", "order_id", "user_id", "from_loc", "item_id"]
LEG_KEY = ["order_id", "user_id", "seq"]


def _entry(e, default_fmt: str) -> tuple[str, str]:
    if isinstance(e, dict):
        return str(e["path"]), e.get("ts_format") or default_fmt
    return str(e), default_fmt


def entry_paths(entries) -> list[str]:
    return [_entry(e, "")[0] for e in entries]


def read_picks(entries, default_fmt: str, window) -> tuple[pd.DataFrame, dict]:
    """Every row of every file in [window[0], window[1]). Returns (events, per-file info)."""
    parts, info = [], {}
    for e in entries:
        path, fmt = _entry(e, default_fmt)
        df = load_wms_picks(path, fmt, window=window)
        # load_wms_picks keeps TRANS_TYPE in transaction_type when the file has it, else the raw TYPE
        is_report = bool((df["transaction_type"].astype("string") == df["type_code"].astype("string")).all())
        info[path] = {"rows": int(len(df)), "ts_format": fmt,
                      "first": str(df["ts"].min()) if len(df) else None,
                      "last": str(df["ts"].max()) if len(df) else None,
                      "format": None if not len(df) else ("pick report" if is_report else "transactions export")}
        if len(df):
            parts.append(df)
    if not parts:
        return pd.DataFrame(), info
    return pd.concat(parts, ignore_index=True), info


def dedup(events: pd.DataFrame) -> pd.DataFrame:
    """Keep the first copy of each transaction (DEDUP_KEY). Two exports of one
    WMS table carry the same transaction number; source_check verifies that on
    every overlap before this is trusted."""
    if events.empty:
        return events
    key = events[DEDUP_KEY].astype("string").fillna("<NA>")
    return events[~key.duplicated()].sort_values(["ts", "event_id"], kind="stable").reset_index(drop=True)


def pick_legs(events: pd.DataFrame, scope: dict, start, end) -> pd.DataFrame:
    """Raw pick-to-pick legs inside [start, end): one row per consecutive pair of
    in-scope picks of one (order, user)."""
    cols = LEG_KEY + ["prev_loc", "loc", "prev_event_id", "event_id", "ts", "seconds"]
    if events.empty:
        return pd.DataFrame(columns=cols)
    s = events[(events["event_class"] == scope["event_class"]) & (events["from_zone"] == scope["zone"])
               & (events["uom"] == scope["uom"]) & (events["ts"] >= start) & (events["ts"] < end)
               & events["order_id"].notna() & events["user_id"].notna()]
    s = s.sort_values(["order_id", "user_id", "ts", "event_id"], kind="stable")
    g = s.groupby(["order_id", "user_id"], sort=False)
    out = pd.DataFrame({
        "order_id": s["order_id"].astype(str), "user_id": s["user_id"].astype(str),
        "seq": g.cumcount() - 1,
        "prev_loc": g["from_loc"].shift().astype("string"), "loc": s["from_loc"].astype("string"),
        "prev_event_id": g["event_id"].shift(), "event_id": s["event_id"],
        "ts": s["ts"], "seconds": g["ts"].diff().dt.total_seconds()})
    return out[out["seq"] >= 0][cols].reset_index(drop=True)


def compare_legs(a: pd.DataFrame, b: pd.DataFrame, tol: dict) -> dict:
    """Do two sides see the same legs? Matched on (order, user, leg number) with
    the same two locations."""
    m = a.merge(b, on=LEG_KEY + ["prev_loc", "loc"], how="inner", suffixes=("_a", "_b"))
    na, nb, nm = len(a), len(b), len(m)
    d = (m["seconds_a"] - m["seconds_b"]).abs()
    same_ids = ((m["event_id_a"] == m["event_id_b"]) & (m["prev_event_id_a"] == m["prev_event_id_b"]))
    r = {
        "legs_a": na, "legs_b": nb, "matched": nm,
        "count_rel_diff": abs(na - nb) / max(na, nb, 1),
        "matched_share_a": nm / na if na else 1.0, "matched_share_b": nm / nb if nb else 1.0,
        "within_seconds_share": float((d <= tol["seconds"]).mean()) if nm else 1.0,
        "abs_seconds_median": float(d.median()) if nm else 0.0,
        "abs_seconds_p99": float(d.quantile(0.99)) if nm else 0.0,
        "abs_seconds_max": float(d.max()) if nm else 0.0,
        "same_event_id_share": float(same_ids.mean()) if nm else 1.0,
    }
    fails = []
    if not (na and nb):
        fails.append(f"no legs on one side (a {na}, b {nb})")
    if r["count_rel_diff"] > tol["count_rel"]:
        fails.append(f"leg counts differ by {r['count_rel_diff']:.4%} (a {na}, b {nb}; tolerance {tol['count_rel']:.4%})")
    for side in ("a", "b"):
        if r[f"matched_share_{side}"] < tol["matched_share"]:
            fails.append(f"{r[f'matched_share_{side}']:.4%} of side {side}'s legs found on the other side "
                         f"(tolerance {tol['matched_share']:.4%})")
    if r["within_seconds_share"] < tol["within_seconds_share"]:
        fails.append(f"{r['within_seconds_share']:.4%} of matched legs agree within {tol['seconds']} s "
                     f"(tolerance {tol['within_seconds_share']:.4%}; max diff {r['abs_seconds_max']:.0f} s)")
    if r["same_event_id_share"] < tol["same_event_id_share"]:
        fails.append(f"{r['same_event_id_share']:.4%} of matched legs carry the same transaction numbers "
                     f"(tolerance {tol['same_event_id_share']:.4%}): deduplication by NO is not safe")
    r["failures"] = fails
    r["ok"] = not fails
    return r


def _span(df: pd.DataFrame):
    return (df["ts"].min(), df["ts"].max()) if len(df) else (None, None)


def check_overlap(ov: dict, default_fmt: str, scope: dict, tol: dict) -> dict:
    """One configured overlap: side a vs side b, and the deduplicated union vs a."""
    start, end = pd.Timestamp(ov["start"]), pd.Timestamp(ov["end"])
    ea, ia = read_picks(ov["a"], default_fmt, (start, end))
    eb, ib = read_picks(ov["b"], default_fmt, (start, end))
    (a0, a1), (b0, b1) = _span(ea), _span(eb)
    out = {"name": ov["name"], "configured": [str(start), str(end)], "files_a": ia, "files_b": ib}
    if a0 is None or b0 is None:
        out.update(ok=False, failures=[f"no rows on side {'a' if a0 is None else 'b'} in {start} .. {end}"])
        return out
    s = max(start, a0, b0)
    e = min(end, a1 + pd.Timedelta(seconds=1), b1 + pd.Timedelta(seconds=1))
    cover = (e - s) / (end - start)
    out["compared"] = [str(s), str(e)]
    out["cover"] = float(cover)
    ea, eb = dedup(ea), dedup(eb)                       # each side's own duplicates (two-part reports)
    la, lb = pick_legs(ea, scope, s, e), pick_legs(eb, scope, s, e)
    ab = compare_legs(la, lb, tol)
    union = compare_legs(la, pick_legs(dedup(pd.concat([ea, eb], ignore_index=True)), scope, s, e), tol)
    fails = [f"a vs b: {f}" for f in ab["failures"]] + [f"deduplicated union vs a: {f}" for f in union["failures"]]
    if cover < tol["min_cover"]:
        fails.insert(0, f"the two sides overlap on only {cover:.1%} of {start} .. {end} "
                        f"(tolerance {tol['min_cover']:.0%})")
    out.update(a_vs_b={k: v for k, v in ab.items() if k != "failures"},
               union_vs_a={k: v for k, v in union.items() if k != "failures"},
               failures=fails, ok=not fails)
    return out


def source_check(checks: dict | None, default_fmt: str, scope: dict) -> dict:
    """Run every configured overlap; raise ValueError naming every failure.
    checks: {tolerance: {...}, overlaps: [{name, a: [...], b: [...], start, end}]}."""
    if not checks or not checks.get("overlaps"):
        return {"overlaps": [], "note": "no overlaps configured"}
    tol = dict(checks["tolerance"])
    res = [check_overlap(ov, default_fmt, scope, tol) for ov in checks["overlaps"]]
    bad = [f"{r['name']}: {f}" for r in res for f in r["failures"]]
    if bad:
        raise ValueError("the exports disagree where they overlap; not planning:\n  " + "\n  ".join(bad))
    return {"tolerance": tol, "overlaps": res}


def check_files(checks: dict | None) -> list[str]:
    if not checks:
        return []
    return sorted({p for ov in checks.get("overlaps", []) for p in entry_paths(ov["a"]) + entry_paths(ov["b"])})
