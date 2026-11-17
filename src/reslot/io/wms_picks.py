"""
Loader for the C&D WMS pick report (yowmspd2_picks_*.csv).

A different export from OneTrack: 18 columns instead of 50, one transaction
type (TYPE 009 = pick), but 30 days instead of 24 minutes. It lands in the
same canonical EVENTS table, so everything that reads events -- including
panel.from_pick_proxy() -- works on it unchanged.

Timestamps are MM/DD/YYYY hh:mm:ss AM/PM. Parsed with an explicit format, not
inferred: the window spans days 15-31, so any day-first misread would fail
loudly rather than silently scramble dates the way the OneTrack Excel export
did.

What this extract does NOT have, and what that means:
  - FROM_EACH: qty_eaches is null. qty is in the row's own UOM (CS or PL).
    panel.from_pick_proxy() sums qty_eaches, so on this source its qty columns
    are zero -- use n_lines (pick count) instead, or convert with an item
    master once one exists.
  - ORD_LINE_NO, LIC, LOAD_ID, lot, warehouse: null.
  - CUST_NO: only CUST_NAME, mapped to customer_id.

The yowmspd2 transactions export (transactions_0915_1015.csv) uses the same
columns plus TRANS_TYPE, the transaction name (PICK, DROPOFF-CP, DROPOFF-FP,
HIVE_PUTAWAY, INBOUND_PUTAWAY, REPLENISHMENT, STARTREPLEN). There TYPE is a
numeric group code (9 pick, 40 dropoff, 8 putaway, 0 replen/misc), not the
pick report's zero-padded "009". When TRANS_TYPE is present:
  - event_class comes from TRANS_TYPE, using the OneTrack taxonomy (same names,
    same classes) plus PICK -> pick. A row with a blank TRANS_TYPE falls back
    to its TYPE.
  - transaction_type holds TRANS_TYPE.
Either way, TYPE is kept as-is in type_code (string, never re-padded).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from ..schemas import EVENTS, validate
from .onetrack import EVENT_CLASS as ONETRACK_EVENT_CLASS
from .onetrack import parse_location

EVENT_CLASS = {"009": "pick"}                      # pick report TYPE codes

# Transactions export TRANS_TYPE names. Same names as OneTrack's
# TRANSACTION_TYPE, so one taxonomy covers both (note OneTrack files
# DROPOFF-CP / DROPOFF-FP under "putaway"; the raw name stays in
# transaction_type to tell them apart).
TRANS_EVENT_CLASS = {**ONETRACK_EVENT_CLASS, "PICK": "pick"}

RENAME = {
    "NO": "event_id",
    "TYPE": "transaction_type",
    "CREATE_DT": "ts",
    "USER_NAME": "user_id",
    "FROM_ITEM": "item_id",
    "FROM_LOCATION": "from_loc",
    "TO_LOCATION": "to_loc",
    "FROM_ZONE": "from_zone",
    "TO_ZONE": "to_zone",
    "FROM_QTY": "qty",
    "FROM_UOM": "uom",
    "ORD_NO": "order_id",
    "CUST_NAME": "customer_id",
}

# Read as strings. ORD_NO carries alpha suffixes on ~0.8% of rows
# (10906764CR, 10905011LR -- likely credits/returns, unconfirmed), and TYPE is
# zero-padded. Numeric inference mangles both.
STRING_COLS = ["TYPE", "TRANS_TYPE", "USER_NAME", "FROM_ITEM", "FROM_LOCATION", "TO_LOCATION",
               "ORD_NO", "CUST_NAME", "FROM_ZONE", "TO_ZONE", "FROM_UOM"]

DEFAULT_TS_FORMAT = "%m/%d/%Y %I:%M:%S %p"


def load_wms_picks(path: str | Path, ts_format: str = DEFAULT_TS_FORMAT, window=None) -> pd.DataFrame:
    """Read a yowmspd2 pick report or transactions export and return
    canonical EVENTS, plus type_code (raw TYPE).

    window: optional (start, end) timestamps. Rows outside [start, end) are dropped
    right after the timestamps are parsed, before the per-row work -- for reading
    one month out of the 365-day pick reports."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        raw = pd.read_parquet(path)
        for c in STRING_COLS:
            if c in raw:
                raw[c] = raw[c].astype("string")
    else:
        raw = pd.read_csv(path, dtype={c: str for c in STRING_COLS}, low_memory=False)

    df = raw.rename(columns=RENAME)

    ts = pd.to_datetime(df["ts"], format=ts_format, errors="coerce")
    bad = int(ts.isna().sum())
    if bad:
        raise ValueError(f"{bad} timestamps did not parse as {ts_format!r}")
    df["ts"] = ts
    if window is not None:
        df = df[(df["ts"] >= window[0]) & (df["ts"] < window[1])].reset_index(drop=True)

    type_code = df["transaction_type"].astype("string")      # raw TYPE, kept as-is
    by_code = type_code.astype(object).map(EVENT_CLASS)
    if "TRANS_TYPE" in df:
        name = df["TRANS_TYPE"].astype("string").str.strip().replace("", pd.NA)
        by_name = name.astype(object).map(TRANS_EVENT_CLASS)
        no_name = name.isna()
        df["event_class"] = by_name.where(~no_name, by_code).fillna("other")
        df["transaction_type"] = name
        unmapped = set(name[~no_name & by_name.isna()]) | set(type_code[no_name & by_code.isna()].dropna())
    else:
        df["event_class"] = by_code.fillna("other")
        unmapped = set(type_code[by_code.isna()].dropna())
    if unmapped:
        warnings.warn(f"Unmapped transaction types -> 'other': {sorted(unmapped)}", stacklevel=2)

    events = validate(df, EVENTS, "events[wms_picks]", strict=False)
    events = events[list(EVENTS)]
    events["type_code"] = type_code
    # Pick reports carry CASES_PER_PALLET per row (item's pallet size); kept for the
    # snapshot-free move cost. Absent in some exports -> NaN.
    events["cases_per_pallet"] = (pd.to_numeric(df["CASES_PER_PALLET"], errors="coerce")
                                  if "CASES_PER_PALLET" in df else float("nan"))

    if len(events):
        locs = events["from_loc"].apply(parse_location).apply(pd.Series)
        events[["from_aisle", "from_bay", "from_level"]] = locs[["aisle", "bay", "level"]]
    else:                         # e.g. a window with no rows in this file
        for c in ("from_aisle", "from_bay", "from_level"):
            events[c] = pd.Series(dtype=object)

    return events.sort_values("ts").reset_index(drop=True)
