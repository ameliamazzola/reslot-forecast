"""Run this against the 5-row C&D sample BEFORE asking for the full pull.

Catches schema surprises while they still cost one email instead of a week.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from reslot.io.demand import load_demand, quality_report

path = sys.argv[1]
df = load_demand(path, strict=False)
print(df.dtypes.to_string(), "\n")
print(pd.Series(quality_report(df)).to_string(), "\n")

missing = [c for c in df.columns if df[c].isna().all()]
if missing:
    print("ALL-NULL COLUMNS -- ask the analyst whether these exist at all:")
    for c in missing:
        print(f"  - {c}")
