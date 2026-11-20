import pandas as pd

src = r"C:\Users\ameli\projects\reslot-forecast\data\raw\Transactions from YOWMSPD2 9.15 thur 10.15 2026.xlsx"
dst = r"C:\Users\ameli\projects\reslot-forecast\data\raw\transactions_0915_1015.csv"
FMT = "%m/%d/%Y %I:%M:%S %p"

sheets = pd.read_excel(src, sheet_name=None)
for name, d in sheets.items():
    print(f"sheet '{name}': {len(d):,} rows")
df = next(iter(sheets.values()))

ts = df["CREATE_DT"]
if not pd.api.types.is_datetime64_any_dtype(ts):
    ts = pd.to_datetime(ts, errors="coerce")
print("unparsed CREATE_DT:", ts.isna().sum())
print("share with non-zero seconds:", round((ts.dt.second != 0).mean(), 3))

for c in df.columns:
    if pd.api.types.is_datetime64_any_dtype(df[c]):
        df[c] = df[c].dt.strftime(FMT)
    elif pd.api.types.is_float_dtype(df[c]) and (df[c].dropna() % 1 == 0).all():
        df[c] = df[c].astype("Int64")
df["CREATE_DT"] = ts.dt.strftime(FMT)

df.to_csv(dst, index=False, encoding="utf-8")
print(f"wrote {len(df):,} rows to {dst}")