# %%
from pathlib import Path
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth

PROJECT_ROOT = Path(r"C:\Users\ameli\projects\reslot-forecast")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
# %%
basket = pd.read_pickle(PROCESSED_DIR / "basket_matrix.pkl")

print("shape:", basket.shape)
print("cell type:", basket.dtypes.iloc[0])

# %%
itemsets = fpgrowth(basket, min_support=0.02, use_colnames=True)

itemsets["length"] = itemsets["itemsets"].apply(len)

print(itemsets["length"].value_counts().sort_index())
print(itemsets.sort_values("support", ascending=False).head(10))
# %%
MIN_SUPPORT = 0.005
MAX_LEN = 2

pairs = fpgrowth(basket, min_support=MIN_SUPPORT, use_colnames=True, max_len=MAX_LEN)
pairs["length"] = pairs["itemsets"].apply(len)

print(pairs["length"].value_counts().sort_index())

# %%
files = ["yowmspd2_archive_picks_past_365days_ranon9.21.26.csv",
         "yowmspd2__picks_past_365days_ranon9.21.26.csv"]

descr = (pd.concat([pd.read_csv(RAW_DIR / f, usecols=["FROM_ITEM", "FROM_DESCR"],
                                dtype={"FROM_ITEM": str}) for f in files])
         .drop_duplicates("FROM_ITEM")
         .set_index("FROM_ITEM")["FROM_DESCR"])

pairs["descriptions"] = pairs["itemsets"].apply(lambda s: [descr.get(sku, "?") for sku in s])
pairs["orders"] = (pairs["support"] * len(basket)).round().astype(int)

top_pairs = pairs[pairs["length"] == 2].sort_values("support", ascending=False)

pd.set_option("display.max_colwidth", 80)
print(top_pairs[["support", "orders", "descriptions"]].head(15).to_string())
# %%
a, b = list(top_pairs.iloc[0]["itemsets"])
both = basket.index[(basket[a] & basket[b]).to_numpy()]

cust = (pd.concat([pd.read_csv(RAW_DIR / f, usecols=["ORD_NO", "CUST_NAME"], dtype={"ORD_NO": str}) for f in files])
        .drop_duplicates("ORD_NO").set_index("ORD_NO")["CUST_NAME"])

print("orders with both:", len(both))
print(cust.reindex(both).value_counts().head(10))

# %%
pairs.to_pickle(PROCESSED_DIR / "frequent_pairs.pkl")
top_pairs[["support", "orders", "descriptions"]].to_csv(PROCESSED_DIR / "top_pairs.csv", index=False)

print("saved", len(pairs), "itemsets")
# %%
