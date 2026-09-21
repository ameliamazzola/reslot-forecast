"""
Loader for the C&D WMS location master (yowmspd2_locations_*.csv).

One row per storage location, 48,741 in the 9/14/26 pull. Most geometry
columns are empty -- X/Y on ~2% of rows, Z / AISLE / RACK / WIDTH / DEPTH not
at all -- which is why WORK_PATH carries the travel model. See docs/decisions.md.

This table, not the transaction log, defines the slot inventory. The 541 pick
faces that saw no activity in the 30-day window exist only here, and they are
the free capacity every re-slotting move lands in.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..schemas import SLOTS, validate
from .onetrack import parse_location

RENAME = {
    "LOCATION": "location",
    "TYPE": "slot_type",
    "WORK_ZONE": "zone",
    "WORK_PATH": "path",
    "PTWY_ZONE": "putaway_zone",
    "PTWY_PATH": "putaway_path",
    "STD_CPCT": "capacity",
    "PICK_UOM": "pick_uom",
}


def load_locations(path: str | Path) -> pd.DataFrame:
    """Read the location master and return canonical SLOTS."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        raw = pd.read_parquet(path)
    else:
        raw = pd.read_csv(path, dtype={"LOCATION": str, "TYPE": str}, low_memory=False)

    dupes = int(raw["LOCATION"].duplicated().sum())
    if dupes:
        raise ValueError(f"location master has {dupes} duplicate LOCATION codes")

    df = raw.rename(columns=RENAME)
    for col in ("path", "putaway_path", "capacity"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["aisle"] = df["location"].map(lambda c: parse_location(c)["aisle"])

    return validate(df, SLOTS, "slots", strict=True)[list(SLOTS)]


def reconcile(events: pd.DataFrame, slots: pd.DataFrame) -> dict:
    """Do the two extracts describe the same building?

    Run before building anything on top of them and record the result in the
    manifest. A perfect match is evidence; assuming one is an assumption.
    """
    seen = set(events["from_loc"].dropna().unique())
    known = set(slots["location"].dropna().unique())
    matched = seen & known
    return {
        "transacted_locations": len(seen),
        "master_locations": len(known),
        "matched": len(matched),
        "match_rate": len(matched) / max(len(seen), 1),
        "missing_examples": sorted(str(m) for m in seen - known)[:20],
    }


def slot_inventory(slots: pd.DataFrame, zone: str, slot_type: str) -> pd.DataFrame:
    """Every slot of a type in a zone, transacted or not."""
    return slots[(slots["zone"] == zone) & (slots["slot_type"] == slot_type)].copy()
