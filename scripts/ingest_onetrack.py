"""Normalise the OneTrack export -> data/interim/events.parquet + a profile.

Run:  python scripts/ingest_onetrack.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reslot.config import load_config
from reslot.io.onetrack import load_onetrack

cfg = load_config()
events = load_onetrack(cfg["data"]["onetrack_path"], cfg["data"]["onetrack_expect_date"])

out = Path(cfg["paths"]["interim"]) / "events.parquet"
out.parent.mkdir(parents=True, exist_ok=True)
events.to_parquet(out, index=False)

print(f"{len(events)} events  {events['ts'].min()} -> {events['ts'].max()}")
print(events["event_class"].value_counts().to_string())
print(f"\nunique items {events['item_id'].nunique()}  "
      f"orders {events['order_id'].nunique()}  users {events['user_id'].nunique()}")
print(f"wrote {out}")
