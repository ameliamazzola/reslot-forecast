"""
Data preprocessing pipeline for SKU-level DAILY demand forecasting.

Turns raw pick-transaction data (one row per pick) into a clean,
feature-rich table: one row per (SKU, calendar day), ready to hand off for
model training.

CHANGED FROM WEEKLY TO DAILY (2026-10-01, per team decision): the team
wants to capture day-of-week effects (e.g. weekends being slower than
weekdays), which a weekly total completely averages away. Grain is now one
row per (SKU, day) instead of one row per (SKU, week). Day-of-week /
weekend features were added specifically to make that pattern visible to
the model (see add_calendar_features below). Everything else - the
CUST_NAME filter, the case-pick/pallet-pull split, the source/caveat
tagging, the fiscal-period split - carries over, just recomputed at daily
instead of weekly resolution.

HOW TO ADD NEW DATA (e.g. next quarter's export from the company):
Just drop the new .csv file into the raw_data/ folder next to this
script, using the same column layout as the existing files, and rerun
`python3 preprocess_demand_data.py`. Nothing in this file needs editing -
it automatically picks up every .csv in raw_data/, filters each one to
real customer-order rows, and recombines everything. The fiscal day/
quarter numbering and the train/test split are also both computed fresh
from whatever data is present, so they extend automatically as more
history accumulates (see add_fiscal_split_labels below).

BACKGROUND (read the project doc "data-finding-real-demand-window.md"
for the full story): the original raw export mixed two transaction
populations - real customer orders and internal/non-customer moves -
distinguishable only by whether CUST_NAME is populated. A second
"archive" file was later found to hold real customer-order history for
months the original file was missing. Both quirks are handled generically
below (the CUST_NAME filter, and combining however many files are
present) rather than as one-off fixes, so this pipeline should not need
another special case the next time a new export shows up - though it's
always worth spot-checking a new file against the assumptions in
load_and_filter() before trusting it blindly.

TEAM ALIGNMENT (added after reviewing a teammate's modeling repo):
  - Column names below are renamed at the very end (finalize_output_schema)
    to the shared schema the rest of the team uses (item_id, date, n_lines,
    source, etc.) so this file can be read by their code without manual
    remapping. "date" in particular now matches their PANEL schema exactly,
    since both are daily grain.
  - Every row is tagged with a "source" column and this file still counts
    as a PROXY for real demand (pick timestamps, not order timestamps),
    the same honesty convention the modeling teammate's code already uses.
    See SOURCE_LABEL / CAVEAT below.
  - Pick transactions mix two different physical operations - case picks
    from the pick face (zone WZPCK) and pallet pulls from reserve storage
    (zones WZGNA-WZGNJ, WZPKG) - that were being silently counted together
    as one "pick_lines" number. They're now also broken out separately
    (case_pick_lines / pallet_pull_lines / other_lines) so the modeling
    teammate can decide which signal(s) to forecast on. The original
    combined total is kept too, as n_lines, so nothing built on it breaks.

Run it as: python3 preprocess_demand_data.py
"""

import glob
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
RAW_DATA_DIR = "raw_data"
RAW_DATA_GLOB = f"{RAW_DATA_DIR}/*.csv"   # picks up every CSV dropped in this folder
OUTPUT_FILE = "sku_day_panel.csv"

KEEP_COLS = ["NO", "CREATE_DT", "FROM_ITEM", "FROM_DESCR", "CASES_PER_PALLET",
             "CUST_NAME", "FROM_ZONE", "FROM_UOM"]

# Which warehouse zone a pick happened in tells you which of two different
# physical operations it was - this isn't in the original file as a clean
# flag, so it's derived from FROM_ZONE. WZPCK is the case-pick face (mostly
# single-case picks that replenish/fulfill from the pick face). The WZGNx /
# WZPKG zones are reserve storage, where whole pallets get pulled instead.
# A pallet pull and a case pick are not the same unit of demand.
CASE_PICK_ZONE = "WZPCK"
PALLET_PULL_ZONES = {"WZGNA", "WZGNB", "WZGNC", "WZGND", "WZGNE", "WZGNF",
                      "WZGNG", "WZGNH", "WZGNJ", "WZPKG"}

# Tag every output row with where this number came from and what it isn't.
SOURCE_LABEL = "pick_proxy_filtered"
CAVEAT = (
    "Every count in this file comes from WAREHOUSE PICK TRANSACTIONS, "
    "filtered to rows with a non-blank CUST_NAME (customer-order picks). "
    "This is a PROXY for real demand, not a direct order-line extract: the "
    "pick timestamp (CREATE_DT) marks when the warehouse released the work, "
    "not when the customer actually placed the order. Replace this source "
    "with a true order-line extract (an order-received timestamp) once one "
    "exists."
)

# Daily lags: yesterday, the last few days, same day last 1-4 weeks, and
# same day last year. lag_7 (this day last week) is one of the single most
# useful features once you're at daily grain, because of day-of-week
# seasonality (see CHANGED FROM WEEKLY note above).
# lag_364 (year-over-year, 52 weeks * 7 days - the fiscal year length used
# throughout this project) will be NaN for almost every row until a second
# fiscal year of data accumulates - it's left in so it starts paying off
# automatically once that happens, with no code change needed.
LAG_DAYS = [1, 2, 3, 7, 14, 21, 28, 364]

# 7/28/56 days = roughly 1/4/8 weeks - the same two "recent average and
# volatility" horizons used in the old weekly version (4 and 8 weeks), plus
# a tight 7-day window since day-level data is noisier day to day than
# week-level totals were.
ROLLING_WINDOWS = [7, 28, 56]

# How many of the most recent complete days to hold out as the final test
# set - one fiscal quarter (13 weeks * 7 days). Stays correct as more
# fiscal years of data are added: it's always "the most recent 91 days,"
# not a fixed calendar date.
TEST_DAYS = 91


def parse_create_dt(raw_dates: pd.Series) -> pd.Series:
    """Parse CREATE_DT, tolerating more than one date style in the same
    column - e.g. "09/21/2025 01:17:59 PM" vs. "9/21/25 13:17". The second
    style is what Excel silently produces if a CSV is ever opened and
    re-saved there, so this is a real risk any time a teammate touches a
    raw file in Excel before handing it off.

    Tries the fast, fixed-format parse first (this is fast: milliseconds
    per 100k rows). Only the rows that don't match it - normally none, or
    a small handful - get re-parsed with the slower "mixed" mode, which
    can infer more than one format but takes ~1-2 seconds per 20k rows,
    too slow to run on 1M+ rows by default.
    """
    parsed = pd.to_datetime(raw_dates, format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    failed = parsed.isna() & raw_dates.notna()
    if failed.any():
        print(f"  {failed.sum():,} of {len(raw_dates):,} dates didn't match the expected "
              f"format - re-parsing those with a slower, more flexible parser "
              f"(likely an Excel re-save changed their format).")
        parsed.loc[failed] = pd.to_datetime(
            raw_dates.loc[failed], format="mixed", dayfirst=False
        )
    return parsed


def load_and_filter(path: str) -> pd.DataFrame:
    """Load one raw file, keep only real customer-order rows.

    A row counts as real demand only if it has a non-blank CUST_NAME -
    this is what separates actual customer pick demand from internal
    warehouse moves (raw-material replenishment, internal transfers,
    etc.). See the project doc for how this was discovered. This filter
    is applied to every file individually, so it's safe even if a future
    export goes back to mixing both transaction types.
    """
    df = pd.read_csv(path, usecols=KEEP_COLS, dtype={"FROM_ITEM": str, "NO": str})
    df["CREATE_DT"] = parse_create_dt(df["CREATE_DT"])

    before = len(df)
    has_customer = df["CUST_NAME"].notna() & (df["CUST_NAME"].str.strip() != "")
    df = df[has_customer].drop(columns=["CUST_NAME"])
    dropped = before - len(df)
    print(f"{path}: dropped {dropped:,} non-customer rows ({dropped/before:.1%} of {before:,}), "
          f"kept {len(df):,}")
    return df


def load_all_raw_files() -> pd.DataFrame:
    """Load and combine every raw export found in raw_data/.

    Drops any transaction ID (NO) seen more than once across all loaded
    files. The two files this pipeline was first built for were verified
    by hand to have zero overlapping transaction IDs - this check is a
    safety net for files added later that haven't been checked the same
    way, so a future overlapping export gets caught here instead of
    silently double-counting demand.
    """
    paths = sorted(glob.glob(RAW_DATA_GLOB))
    if not paths:
        raise FileNotFoundError(
            f"No CSV files found matching {RAW_DATA_GLOB} - "
            f"put the raw exports in {RAW_DATA_DIR}/ before running this script."
        )
    print(f"Found {len(paths)} raw file(s): {[p for p in paths]}")

    frames = [load_and_filter(p) for p in paths]
    combined = pd.concat(frames, ignore_index=True)

    before = len(combined)
    combined = combined.drop_duplicates(subset=["NO"], keep="first")
    dupes = before - len(combined)
    if dupes:
        print(f"WARNING: dropped {dupes:,} rows with a transaction ID (NO) that appeared "
              f"in more than one file - investigate before trusting this run's numbers.")
    else:
        print("No duplicate transaction IDs across files - confirmed safe to combine as-is.")

    return combined.drop(columns=["NO"])


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["CREATE_DT", "FROM_ITEM"])
    df["FROM_DESCR"] = df["FROM_DESCR"].fillna("UNKNOWN")
    df["CASES_PER_PALLET"] = df["CASES_PER_PALLET"].fillna(-1)
    df["FROM_ZONE"] = df["FROM_ZONE"].fillna("UNKNOWN")
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with missing date or item ID ({dropped/before:.2%})")
    return df


def classify_process(zone: str) -> str:
    """Which physical operation a pick zone represents.

    case_pick: the pick face (WZPCK) - mostly single-case picks.
    pallet_pull: reserve storage (WZGNA-WZGNJ, WZPKG) - whole pallets.
    other: everything else (staging, VMS, a handful of rarely-used zones).
    """
    if zone == CASE_PICK_ZONE:
        return "case_pick"
    if zone in PALLET_PULL_ZONES:
        return "pallet_pull"
    return "other"


def build_sku_day_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket into calendar days, count pick lines (total and by process),
    fill the zero grid.

    The raw export doesn't start or end exactly at midnight (the first
    timestamp is mid-morning, the last is mid-day), so the very first and
    very last calendar day in the data only have a few hours of coverage
    each - a fake near-zero "demand" day if left in. Those two boundary
    days are dropped here, automatically, on every run - this isn't
    specific to this year's dates. (This is the same boundary-trimming
    idea used in the old weekly version, just applied to single days
    instead of whole weeks - and it lands on the identical date range:
    2025-09-22 through 2026-09-20, 364 full days = the same 52 full weeks
    as before.)
    """
    df = df.copy()
    df["day"] = df["CREATE_DT"].dt.normalize()
    df["process"] = df["FROM_ZONE"].map(classify_process)

    data_min, data_max = df["CREATE_DT"].min(), df["CREATE_DT"].max()
    first_day, last_day = data_min.normalize(), data_max.normalize()
    full_day_lo = first_day if data_min == first_day else first_day + pd.Timedelta(days=1)
    full_day_hi = last_day if data_max == last_day else last_day - pd.Timedelta(days=1)

    # Total pick lines per (day, SKU) - kept so nothing downstream (lags,
    # rolling features) needs a second definition of volume.
    panel = (
        df.groupby(["day", "FROM_ITEM"])
        .size()
        .rename("pick_lines")
        .reset_index()
    )

    # Breakdown by process. reindex guarantees all three category columns
    # always exist, even if one process is absent from a given file.
    breakdown = (
        df.groupby(["day", "FROM_ITEM", "process"])
        .size()
        .unstack("process", fill_value=0)
        .reindex(columns=["case_pick", "pallet_pull", "other"], fill_value=0)
    )
    breakdown.columns = [f"{c}_lines" for c in breakdown.columns]
    breakdown = breakdown.reset_index()

    panel = panel.merge(breakdown, on=["day", "FROM_ITEM"], how="left")

    all_days_seen = panel["day"].unique()
    full_days = [d for d in all_days_seen if full_day_lo <= d <= full_day_hi]
    dropped_days = sorted(set(all_days_seen) - set(full_days))
    if dropped_days:
        print(f"Dropped {len(dropped_days)} partial boundary day(s) from the panel: "
              f"{[str(d.date()) for d in dropped_days]}")

    all_skus = panel["FROM_ITEM"].unique()
    full_grid = pd.date_range(min(full_days), max(full_days), freq="D")
    full_grid = pd.MultiIndex.from_product(
        [full_grid, all_skus], names=["day", "FROM_ITEM"]
    ).to_frame(index=False)

    panel = full_grid.merge(panel, on=["day", "FROM_ITEM"], how="left")
    count_cols = ["pick_lines", "case_pick_lines", "pallet_pull_lines", "other_lines"]
    panel[count_cols] = panel[count_cols].fillna(0)
    return panel.sort_values(["FROM_ITEM", "day"]).reset_index(drop=True)


def build_sku_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """One row per SKU: static attributes that don't change day to day."""
    attrs = (
        df.groupby("FROM_ITEM")
        .agg(
            FROM_DESCR=("FROM_DESCR", lambda x: x.mode().iloc[0]),
            CASES_PER_PALLET=("CASES_PER_PALLET", lambda x: x.mode().iloc[0]),
        )
        .reset_index()
    )
    # First word of the description is a brand code in this data (e.g.
    # "TROJ"=Trojan, "AHLLD"=Arm & Hammer Liquid Laundry Detergent,
    # "ORAJEL", "NAIR", "ZICAM") - a reasonable grouping for pooling
    # similar SKUs, though some brand groups have only 1-2 SKUs in them.
    attrs["product_family"] = attrs["FROM_DESCR"].str.split().str[0]

    # Velocity tier (A/B/C), based on each SKU's share of total pick
    # volume over the whole window. Standard ABC/Pareto cutoffs (80%/95%
    # of cumulative volume) - the same convention independently used in
    # the modeling teammate's repo.
    total_by_sku = df.groupby("FROM_ITEM").size().sort_values(ascending=False)
    cum_share = total_by_sku.cumsum() / total_by_sku.sum()
    tier = pd.cut(cum_share, bins=[-0.01, 0.80, 0.95, 1.0], labels=["A", "B", "C"])
    attrs = attrs.merge(
        tier.rename("velocity_tier").reset_index().rename(columns={"index": "FROM_ITEM"}),
        on="FROM_ITEM", how="left",
    )
    return attrs


def add_lag_and_rolling_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-SKU history: what this SKU did recently.

    shift(1) is applied before every rolling calculation, so "today" is
    never included in its own rolling average or lag - that would be
    leakage (using the answer to help predict the answer).
    """
    panel = panel.sort_values(["FROM_ITEM", "day"]).reset_index(drop=True)
    g = panel.groupby("FROM_ITEM")["pick_lines"]

    for lag in LAG_DAYS:
        panel[f"lag_{lag}"] = g.shift(lag)

    for window in ROLLING_WINDOWS:
        shifted = g.shift(1)
        panel[f"roll_mean_{window}"] = (
            shifted.groupby(panel["FROM_ITEM"]).rolling(window).mean().reset_index(level=0, drop=True)
        )
        panel[f"roll_std_{window}"] = (
            shifted.groupby(panel["FROM_ITEM"]).rolling(window).std().reset_index(level=0, drop=True)
        )

    # Coefficient of variation on the 56-day window (the ~8-week horizon
    # used for this in the old weekly version): how volatile this SKU's
    # demand has been lately, relative to its own average.
    panel["demand_cv_56"] = panel["roll_std_56"] / panel["roll_mean_56"].replace(0, np.nan)

    # Days since this SKU was last actually picked - a direct measure of
    # "how cold is this SKU right now."
    is_active = (panel["pick_lines"] > 0).astype(int)
    active_day_idx = panel["day"].map(pd.Timestamp.toordinal).to_numpy()
    panel["_active_day_num"] = np.where(is_active == 1, active_day_idx, np.nan)
    panel["_last_active_day_num"] = (
        panel.groupby("FROM_ITEM")["_active_day_num"].apply(lambda s: s.shift(1).ffill())
        .reset_index(level=0, drop=True)
    )
    panel["days_since_last_pick"] = active_day_idx - panel["_last_active_day_num"]
    panel = panel.drop(columns=["_active_day_num", "_last_active_day_num"])

    return panel


def add_calendar_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Calendar features, now including day-of-week / weekend - the whole
    reason this pipeline moved from weekly to daily grain. A weekly total
    makes every day inside that week indistinguishable; at daily grain the
    model can actually learn "Saturdays and Sundays are slower.\""""
    day = panel["day"]
    panel["day_of_week"] = day.dt.dayofweek          # Monday=0 ... Sunday=6
    panel["day_name"] = day.dt.day_name()
    panel["is_weekend"] = day.dt.dayofweek.isin([5, 6]).astype(int)

    panel["week_of_year"] = day.dt.isocalendar().week.astype(int)
    panel["month"] = day.dt.month
    panel["quarter"] = day.dt.quarter
    panel["is_holiday_season"] = panel["month"].isin([11, 12]).astype(int)

    # Exact holiday dates - more precise than the old weekly version's
    # "whole week containing the holiday" flag, since daily grain lets us
    # mark the specific day. Extend this dict by hand each year new dates
    # are known - there's no reliable way to compute "US Thanksgiving"
    # purely from a formula without a holiday-calendar library.
    us_holidays = {
        "thanksgiving_day": [pd.Timestamp("2025-11-27"), pd.Timestamp("2026-11-26")],
        "christmas_day": [pd.Timestamp("2025-12-25"), pd.Timestamp("2026-12-25")],
    }
    for name, dates in us_holidays.items():
        panel[f"is_{name}"] = day.isin(dates).astype(int)

    return panel


def add_fiscal_split_labels(panel: pd.DataFrame) -> pd.DataFrame:
    """Label each day with its position in the fiscal calendar and the
    train/test split this implies.

    fiscal_day counts sequentially from the first full day present in
    whatever data is loaded (day 1 is always "the earliest full day this
    run has", not a hardcoded date) - so this keeps working exactly the
    same way once a second, third, etc. fiscal year of data exists.
    fiscal_week/fiscal_quarter/fiscal_year assume a standard 364-day
    (52-week) fiscal year, matching the rest of the project.

    split is always "the most recent TEST_DAYS complete days in the data
    are the test set, everything else is train" - the same one-fiscal-
    quarter holdout used in the weekly version, just expressed in days
    (91 days = 13 weeks) instead of weeks.
    """
    day_order = sorted(panel["day"].unique())
    n_days = len(day_order)
    fiscal_day_num = {d: i + 1 for i, d in enumerate(day_order)}
    panel["fiscal_day"] = panel["day"].map(fiscal_day_num)
    panel["fiscal_week"] = ((panel["fiscal_day"] - 1) // 7) + 1
    panel["fiscal_year"] = ((panel["fiscal_day"] - 1) // 364) + 1
    panel["fiscal_quarter"] = (((panel["fiscal_day"] - 1) % 364) // 91) + 1
    panel["split"] = np.where(panel["fiscal_day"] > n_days - TEST_DAYS, "test", "train")
    return panel


def finalize_output_schema(panel: pd.DataFrame) -> pd.DataFrame:
    """Rename to the column names the rest of the team's code already uses,
    and tag every row with where this number came from.

    "day" is renamed to "date" here specifically because, now that this
    file is daily grain, that's an exact match for the modeling teammate's
    PANEL schema (item_id, date, ...) - no more translation needed between
    the two.
    """
    panel = panel.rename(columns={
        "FROM_ITEM": "item_id",
        "FROM_DESCR": "item_descr",
        "CASES_PER_PALLET": "cases_per_pallet",
        "pick_lines": "n_lines",
        "day": "date",
    })

    panel["date"] = panel["date"].dt.date.astype(str)
    panel["source"] = SOURCE_LABEL

    lead_cols = ["item_id", "date", "day_of_week", "day_name", "is_weekend", "source",
                 "n_lines", "case_pick_lines", "pallet_pull_lines", "other_lines"]
    other_cols = [c for c in panel.columns if c not in lead_cols]
    return panel[lead_cols + other_cols]


def run_pipeline() -> pd.DataFrame:
    combined = load_all_raw_files()
    combined = clean_raw(combined)

    panel = build_sku_day_panel(combined)
    attrs = build_sku_attributes(combined)
    panel = panel.merge(attrs, on="FROM_ITEM", how="left")

    panel = add_lag_and_rolling_features(panel)
    panel = add_calendar_features(panel)
    panel = add_fiscal_split_labels(panel)
    panel = finalize_output_schema(panel)

    return panel


if __name__ == "__main__":
    final_panel = run_pipeline()
    final_panel.to_csv(OUTPUT_FILE, index=False)
    n_days = final_panel["date"].nunique()
    n_test_days = final_panel.loc[final_panel["split"] == "test", "date"].nunique()
    print(f"\nSaved {len(final_panel):,} rows, {final_panel.shape[1]} columns to {OUTPUT_FILE}")
    print(f"Days covered: {n_days} ({n_days - n_test_days} train / {n_test_days} test) "
          f"| SKUs: {final_panel['item_id'].nunique()}")
    print(f"\nSource: {SOURCE_LABEL}\n{CAVEAT}")
    print()
    print(final_panel.head(3).to_string())
