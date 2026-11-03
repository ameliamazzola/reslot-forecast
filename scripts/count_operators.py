import sys; sys.path.insert(0, "src")
import pandas as pd
from reslot.io.wms_picks import load_wms_picks

path = r"C:\Users\ameli\projects\reslot-forecast\data\raw\transactions_0915_1015.csv"
ev = load_wms_picks(path)
print("rows:", len(ev), "| dates:", ev.ts.min(), "to", ev.ts.max())
print("event_class:", ev.event_class.value_counts(dropna=False).to_dict())

w = ev[(ev.event_class == "pick") & (ev.ts >= "2026-10-05") & (ev.ts < "2026-10-12")]
print("picks by uom:", w.uom.value_counts().to_dict())
print("PL picks by zone:", w[w.uom == "PL"].from_zone.value_counts().head(10).to_dict())

case = w[(w.from_zone == "WZPCK") & (w.uom == "CS")]
pallet = w[(w.uom == "PL") & (w.from_zone != "WZPKG")]
groups = {"case": case, "pallet": pallet, "combined": pd.concat([case, pallet])}

for name, p in groups.items():
    hourly = p.groupby(p.ts.dt.floor("h"))["user_id"].nunique()
    print(f"{name}: week users {p.user_id.nunique()} | median/hr {hourly.median()} "
          f"| p90 {hourly.quantile(0.9)} | max {hourly.max()}")