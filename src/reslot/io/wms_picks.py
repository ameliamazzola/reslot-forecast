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
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from ..schemas import EVENTS, validate
from .onetrack import parse_location

EVENT_CLASS = {"009": "pick"}

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
STRING_COLS = ["TYPE", "USER_NAME", "FROM_ITEM", "FROM_LOCATION", "TO_LOCATION",
               "ORD_NO", "CUST_NAME", "FROM_ZONE", "TO_ZONE", "FROM_UOM"]

DEFAULT_TS_FORMAT = "%m/%d/%Y %I:%M:%S %p"


def load_wms_picks(path: str | Path, ts_format: str = DEFAULT_TS_FORMAT) -> pd.DataFrame:
    """Read a yowmspd2 pick report and return canonical EVENTS."""
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

    df["event_class"] = df["transaction_type"].map(EVENT_CLASS).fillna("other")
    unmapped = sorted(set(df.loc[df["event_class"] == "other", "transaction_type"].dropna()))
    if unmapped:
        warnings.warn(f"Unmapped transaction types -> 'other': {unmapped}", stacklevel=2)

    events = validate(df, EVENTS, "events[wms_picks]", strict=False)
    events = events[list(EVENTS)]

    locs = events["from_loc"].apply(parse_location).apply(pd.Series)
    events[["from_aisle", "from_bay", "from_level"]] = locs[["aisle", "bay", "level"]]

    return events.sort_values("ts").reset_index(drop=True)
