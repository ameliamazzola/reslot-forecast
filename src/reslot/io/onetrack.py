"""
Loader for the C&D OneTrack movement export (1004WMS_ONETRACK_*.xlsx).

This is the EXECUTION log, not demand. See docs/data-notes.md. We load it
because it gives us travel behaviour, zone workload, replenishment volume and
the incumbent slotting -- all of which the simulator needs. It is NOT the
forecaster's input, except via the explicitly-labelled pick_proxy path.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd

from ..schemas import EVENTS, validate

# --------------------------------------------------------------------------
# Transaction taxonomy
# --------------------------------------------------------------------------
# Raw TRANSACTION_TYPE values seen in the 2026-09-10 sample. Anything not
# listed lands in "other" and is logged -- update this map as new types appear
# in a longer extract rather than letting them be silently dropped.
EVENT_CLASS = {
    "CASE_PICK": "pick",
    "FULL_PALLET_PICK": "pick",
    "WO_PICK": "pick",
    "INBOUND_PUTAWAY": "putaway",
    "HIVE_PUTAWAY": "putaway",
    "DROPOFF-FP": "putaway",
    "DROPOFF-CP": "putaway",
    "DROPOFF_EXC_HIVE": "putaway",
    "DROPOFF_EXC_INBOUND": "putaway",
    "REPLENISHMENT": "replen",
    "STARTREPLEN": "replen",
    "LOAD": "ship",
    "SHIP": "ship",
    "CONTAINER_OUT": "ship",
    "DOOR-TO-DKMOVE": "stage",
    "DK-TO-YARD": "stage",
    "YARD-TO-DKMOVE": "stage",
    "YARD-TO-DOCK": "stage",
    "YARD-TO-YARD": "stage",
    "LYSTAGE-TO-PLSTAGE": "stage",
    "GENWHSE-TO-VMS": "internal_move",
    "TRANSFER": "internal_move",
    "LLD-CAT-MOVE": "internal_move",
    "MOVE_TO_DAMAGE": "internal_move",
    "MOVE_OUTOF_DAMAGE": "internal_move",
    "COUNTBACK": "count",
    "ADJUST": "adjust",
    "PRODUCTION": "production",
    "YARD_CHECKIN": "admin",
    "CHECKIN_RECEIVING": "admin",
    "LOG_ON": "admin",
    "LOG_IN": "admin",
    "LOG_OFF": "admin",
    "EDIT_PL_LOT_DT": "admin",
    "ALIAS_LICENSE": "admin",
    "UNALLOCATEORDERUI": "admin",
}

RENAME = {
    "NO": "event_id",
    "CREATE_DT": "ts",
    "TRANSACTION_TYPE": "transaction_type",
    "WORK_TYPE": "work_type",
    "USER_NAME": "user_id",
    "FROM_ITEM": "item_id",
    "FROM_LOCATION": "from_loc",
    "TO_LOCATION": "to_loc",
    "FROM_ZONE": "from_zone",
    "TO_ZONE": "to_zone",
    "FROM_QTY": "qty",
    "FROM_UOM": "uom",
    "FROM_EACH": "qty_eaches",
    "FROM_LIC": "lic",
    "ORD_NO": "order_id",
    "ORD_LINE_NO": "order_line",
    "CUST_NO": "customer_id",
    "LOAD_ID": "load_id",
    "FROM_LOT_NUMBER": "lot",
    "FROM_WAREHOUSE": "warehouse",
}

# Pick-face codes look like KK038A / GG244A / D161E / Q041B
# -> aisle letters, bay digits, level letter. CONFIRM WITH THE WMS ANALYST
# before any travel-cost model depends on this.
SLOT_RE = re.compile(r"^(?P<aisle>[A-Z]{1,2})(?P<bay>\d{2,3})(?P<level>[A-Z])$")


def parse_location(code: str | None) -> dict:
    """Decompose a location code. Returns nulls for non-rack locations
    (RT-xx reserve, PL-xx staging, DK-xx dock, V-xxxx vehicle)."""
    if not isinstance(code, str):
        return {"aisle": None, "bay": None, "level": None}
    m = SLOT_RE.match(code.strip().upper())
    if not m:
        return {"aisle": None, "bay": None, "level": None}
    return {"aisle": m["aisle"], "bay": int(m["bay"]), "level": m["level"]}


def load_onetrack(path: str | Path, expect_date: str | None = None) -> pd.DataFrame:
    """Read a OneTrack export and return canonical EVENTS.

    expect_date: 'YYYY-MM-DD' parsed from the filename stamp. Excel exports of
    this report have shown day-first / month-first ambiguity (file stamped
    260910 contained cells reading 2026-10-09). Pass it and we warn on
    mismatch instead of silently forecasting on scrambled dates.
    """
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        raw = pd.read_excel(path)
    else:
        raw = pd.read_csv(path)

    df = raw.rename(columns=RENAME)

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    if expect_date is not None:
        observed = df["ts"].dt.date.mode()
        if len(observed) and str(observed.iloc[0]) != expect_date:
            warnings.warn(
                f"Date mismatch: filename implies {expect_date}, cells say "
                f"{observed.iloc[0]}. Likely DD/MM vs MM/DD. Confirm the export "
                f"locale with the WMS analyst before using these timestamps.",
                stacklevel=2,
            )

    df["event_class"] = df["transaction_type"].map(EVENT_CLASS).fillna("other")
    unmapped = sorted(set(df.loc[df["event_class"] == "other", "transaction_type"].dropna()))
    if unmapped:
        warnings.warn(f"Unmapped transaction types -> 'other': {unmapped}", stacklevel=2)

    for col in ("item_id", "order_id", "order_line", "customer_id", "lic", "load_id"):
        if col in df:
            df[col] = df[col].apply(
                lambda v: None if pd.isna(v)
                else (str(int(v)) if isinstance(v, float) and float(v).is_integer() else str(v))
            )

    events = validate(df, EVENTS, "events", strict=False)
    # Drop raw columns outside the contract. They are recoverable from the
    # source file; keeping them here invites mixed-dtype breakage downstream.
    events = events[list(EVENTS)]

    locs = events["from_loc"].apply(parse_location).apply(pd.Series)
    events[["from_aisle", "from_bay", "from_level"]] = locs[["aisle", "bay", "level"]]

    return events.sort_values("ts").reset_index(drop=True)
