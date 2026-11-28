import pandas as pd

path = r"C:\Users\ameli\projects\reslot-forecast\data\raw\transactions_0915_1015.csv"
df = pd.read_csv(path, dtype=str, keep_default_na=False)

print(pd.crosstab(df.TRANS_TYPE, df.TYPE), "\n")
print(pd.crosstab(df.TRANS_TYPE, df.FROM_UOM), "\n")
for code in sorted(df.TRANS_TYPE.unique()):
    sub = df[df.TRANS_TYPE == code]
    print(f"code {code}: {len(sub):,} rows | top FROM_ZONE {sub.FROM_ZONE.value_counts().head(4).to_dict()} "
          f"| top TO_ZONE {sub.TO_ZONE.value_counts().head(4).to_dict()} | has ORD_NO {round((sub.ORD_NO != '').mean(), 2)}")