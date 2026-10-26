# %%
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\ameli\projects\reslot-forecast")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

labels = pd.read_csv(PROCESSED_DIR / "sku_labels.csv", dtype={"sku": str})
fc = pd.read_csv(PROCESSED_DIR / "forecast output" / "pick_count_forecast_outputs" / "forecast_sku_next_28d_picks.csv", dtype={"item_id": str})

labels = labels.rename(columns={"group": "affinity_group"})
fc = fc.rename(columns={"item_id": "sku", "group": "demand_group"})

keep = ["sku", "demand_group", "tier", "expected_picks_28d", "last28_picks",
        "change_flag", "picks_low_80", "picks_high_80", "case_share_28d"]
combined = labels.merge(fc[keep], on="sku", how="left", indicator=True)

print(combined["_merge"].value_counts())
print("forecast SKUs not in affinity labels:", len(set(fc["sku"]) - set(labels["sku"])))
combined = combined.drop(columns="_merge")
# %%
print(combined[combined["demand_group"].isna()][["sku", "descr", "affinity_group"]].to_string(index=False))

# %%
clustered = combined[combined["cluster"] >= 0]

print(pd.crosstab(clustered["cluster"], clustered["demand_group"], margins=True), "\n")
print(pd.crosstab(combined["affinity_group"], combined["demand_group"]))
# %%
print(pd.crosstab(clustered["cluster"], clustered["change_flag"]), "\n")

candidates = combined[(combined["change_flag"] == "up") &
                      (combined["demand_group"].isin(["Steady", "Variable"]))]

print("anticipatory candidates:", len(candidates))
print(candidates[["sku", "descr", "cluster", "demand_group", "last28_picks",
                  "expected_picks_28d", "picks_low_80", "picks_high_80"]]
      .sort_values("expected_picks_28d", ascending=False).to_string(index=False))

combined.to_csv(PROCESSED_DIR / "sku_combined_labels.csv", index=False)
candidates.to_csv(PROCESSED_DIR / "anticipatory_candidates.csv", index=False)