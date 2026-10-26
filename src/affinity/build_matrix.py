# %%
from pathlib import Path
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

PROJECT_ROOT = Path(r"C:\Users\ameli\projects\reslot-forecast")
DATA_DIR = PROJECT_ROOT / "data" / "raw"

ARCHIVE_FILE = DATA_DIR / "yowmspd2_archive_picks_past_365days_ranon9.21.26.csv"
CURRENT_FILE = DATA_DIR / "yowmspd2__picks_past_365days_ranon9.21.26.csv"
print(ARCHIVE_FILE.exists(), CURRENT_FILE.exists())

# %%
dtypes = {"ORD_NO": str, "FROM_ITEM": str}

archive = pd.read_csv(ARCHIVE_FILE, dtype=dtypes)
current = pd.read_csv(CURRENT_FILE, dtype=dtypes)

print("archive:", archive.shape)
print("current:", current.shape)

# %%
picks = pd.concat([archive, current], ignore_index=True)

picks["CREATE_DT"] = pd.to_datetime(picks["CREATE_DT"], format="%m/%d/%Y %I:%M:%S %p")

print("combined:", picks.shape)
print("NO unique:", picks["NO"].is_unique)
print("date range:", picks["CREATE_DT"].min(), "to", picks["CREATE_DT"].max())

# %%
print(picks["FROM_UOM"].value_counts(), "\n")
print(picks["FROM_ZONE"].value_counts(), "\n")
print("blank customers:", picks["CUST_NAME"].isna().sum())

skus_per_order = picks.groupby("ORD_NO")["FROM_ITEM"].nunique()
print("orders:", len(skus_per_order))
print("single-SKU orders:", (skus_per_order == 1).sum())

# %%
vitamin_pattern = "VITFUS|VITAMIN|CRITTER|GUMM"
vit = picks[picks["FROM_DESCR"].str.contains(vitamin_pattern, case=False, na=False)]
print(vit[["FROM_ITEM", "FROM_DESCR", "FROM_ZONE"]].drop_duplicates("FROM_ITEM").to_string())
# %%
vms_skus = picks.loc[picks["FROM_ZONE"] == "WZVMS", "FROM_ITEM"].unique()

vms_info = (picks[picks["FROM_ITEM"].isin(vms_skus)]
            .groupby(["FROM_ITEM", "FROM_DESCR"])["FROM_ZONE"]
            .unique())
print(vms_info.to_string())

# %%
# ---------------- FILTER SWITCHES ----------------
DROP_VMS               = True           # vitamins discontinued
DROP_NON_RACK_ZONES    = False          # confirm with site before turning on
NON_RACK_ZONES         = ["WZPKG", "WZNDK"]
DROP_LETTER_ORDERS     = True          # internal orders like 10852008A, no customer
DROP_INACTIVE_SKUS     = True          # SKUs not picked since the cutoff
INACTIVE_CUTOFF        = "2026-03-01"
DROP_RARE_SKUS         = True          # SKUs on very few orders
MIN_ORDERS_PER_SKU     = 5
DROP_VITAMIN_BRANDS = True
VITAMIN_PATTERN = "VITFUS|CRITTER"     # adjust after Step 1
DROP_LARGE_ORDERS      = False          # orders with a huge number of SKUs
MAX_SKUS_PER_ORDER     = 100
DROP_SINGLE_SKU_ORDERS = False          # always runs last
# -------------------------------------------------

filter_log = [("start", len(picks), picks["ORD_NO"].nunique(), picks["FROM_ITEM"].nunique())]

def log_step(name):
    filter_log.append((name, len(picks), picks["ORD_NO"].nunique(), picks["FROM_ITEM"].nunique()))

# 1. VMS: drop every SKU ever picked from the VMS zone, in any zone
if DROP_VMS:
    vms_skus = picks.loc[picks["FROM_ZONE"] == "WZVMS", "FROM_ITEM"].unique()
    picks = picks[~picks["FROM_ITEM"].isin(vms_skus)]
    log_step("drop VMS SKUs")

if DROP_VITAMIN_BRANDS:
    vit_skus = picks.loc[picks["FROM_DESCR"].str.contains(VITAMIN_PATTERN, case=False, na=False), "FROM_ITEM"].unique()
    picks = picks[~picks["FROM_ITEM"].isin(vit_skus)]
    log_step("drop vitamin brands")

# 2. Non-rack zones
if DROP_NON_RACK_ZONES:
    picks = picks[~picks["FROM_ZONE"].isin(NON_RACK_ZONES)]
    log_step("drop non-rack zones")

# 3. Letter-suffix (internal) orders
if DROP_LETTER_ORDERS:
    picks = picks[picks["ORD_NO"].str.fullmatch(r"\d+")]
    log_step("drop letter orders")

# 4. Inactive SKUs
if DROP_INACTIVE_SKUS:
    last_pick = picks.groupby("FROM_ITEM")["CREATE_DT"].max()
    active_skus = last_pick[last_pick >= INACTIVE_CUTOFF].index
    picks = picks[picks["FROM_ITEM"].isin(active_skus)]
    log_step("drop inactive SKUs")

# 5. Rare SKUs
if DROP_RARE_SKUS:
    orders_per_sku = picks.groupby("FROM_ITEM")["ORD_NO"].nunique()
    common_skus = orders_per_sku[orders_per_sku >= MIN_ORDERS_PER_SKU].index
    picks = picks[picks["FROM_ITEM"].isin(common_skus)]
    log_step("drop rare SKUs")

# 6. Very large orders
if DROP_LARGE_ORDERS:
    skus_per_order = picks.groupby("ORD_NO")["FROM_ITEM"].nunique()
    normal_orders = skus_per_order[skus_per_order <= MAX_SKUS_PER_ORDER].index
    picks = picks[picks["ORD_NO"].isin(normal_orders)]
    log_step("drop large orders")

# 7. Single-SKU orders (last, since earlier filters can create new ones)
if DROP_SINGLE_SKU_ORDERS:
    skus_per_order = picks.groupby("ORD_NO")["FROM_ITEM"].nunique()
    multi_sku_orders = skus_per_order[skus_per_order >= 2].index
    picks = picks[picks["ORD_NO"].isin(multi_sku_orders)]
    log_step("drop single-SKU orders")

filter_table = pd.DataFrame(filter_log, columns=["step", "rows", "orders", "skus"])
filter_table["rows_removed"] = filter_table["rows"].diff().fillna(0).astype(int) * -1
print(filter_table.to_string(index=False))
# %%
orders = picks.groupby("ORD_NO")["FROM_ITEM"].unique()

dataset = [list(skus) for skus in orders]
order_ids = orders.index.tolist()

print("transactions:", len(dataset))
print(dataset[:3])
# %%
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset, sparse=True)

print("shape:", te_ary.shape)
print("number of SKUs:", len(te.columns_))
# %%
basket = pd.DataFrame.sparse.from_spmatrix(te_ary, index=order_ids, columns=te.columns_)

print(basket.iloc[:5, :5])
print("density:", basket.sparse.density)
# %%
# Check 1: decode the first 3 rows back into lists and compare to the originals
print(te.inverse_transform(te_ary[:3].toarray()))
print([sorted(x) for x in dataset[:3]])

# Check 2: one order, raw data vs matrix
oid = order_ids[689]
raw_skus = set(picks.loc[picks["ORD_NO"] == oid, "FROM_ITEM"])
row = basket.loc[oid]
matrix_skus = set(row[row].index)
print("order", oid, "matches:", raw_skus == matrix_skus)
# %%
out_dir = PROJECT_ROOT / "data" / "processed"
out_dir.mkdir(parents=True, exist_ok=True)

basket.to_pickle(out_dir / "basket_matrix.pkl")
print("saved to", out_dir)

# %%
