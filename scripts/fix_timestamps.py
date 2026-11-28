import pandas as pd

path = r"C:\Users\ameli\projects\reslot-forecast\data\raw\Transactions from YOWMSPD2 9.15 thur 10.15 2026.csv"
FMT = "%m/%d/%Y %I:%M:%S %p"

df = pd.read_csv(path, dtype=str, keep_default_na=False)

for c in df.columns:
    vals = df[c][df[c] != ""]
    if vals.empty:
        continue
    parsed = pd.to_datetime(vals, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    if parsed.notna().mean() > 0.99:          # column is a timestamp column
        df.loc[vals.index, c] = parsed.dt.strftime(FMT)
        print(f"fixed '{c}': e.g. {df.loc[vals.index[0], c]}  ({parsed.isna().sum()} unparsed)")

df.to_csv(path, index=False, encoding="utf-8")
print("saved", path)