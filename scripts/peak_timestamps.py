import pandas as pd

path = r"C:\Users\ameli\projects\reslot-forecast\data\raw\transactions_0915_1015.csv"
df = pd.read_csv(path, dtype=str, nrows=5)

print("columns:", list(df.columns))
for c in df.columns:
    if any(k in c.upper() for k in ("DATE", "TIME", "TS", "DT")):
        print(f"{c}: {df[c].tolist()}")