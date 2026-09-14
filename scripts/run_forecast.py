"""Build the panel, backtest every model, write results + manifest.

Run:  python scripts/run_forecast.py [configs/default.yaml]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from reslot.config import load_config
from reslot.features import panel as P
from reslot.forecast.baselines import Croston, MovingAverage, SeasonalNaive
from reslot.forecast.evaluate import rolling_origin
from reslot.forecast.sarima import SarimaPerItem
from reslot.utils.manifest import write_manifest

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml")

# --- demand in ------------------------------------------------------------
if cfg["data"]["source"] == "synthetic":
    from reslot.io.synthetic import generate_demand
    demand = generate_demand(seed=cfg["seed"], **cfg["synthetic"])
else:
    from reslot.io.demand import load_demand, quality_report
    demand = load_demand(cfg["data"]["demand_path"], strict=True)
    print(pd.Series(quality_report(demand)).to_string())

# --- panel ----------------------------------------------------------------
panel = P.densify(P.from_demand(demand, cfg["panel"]["include_cancelled"]))
inter = P.intermittency(panel)
print(f"panel: {panel['item_id'].nunique()} items x "
      f"{panel['date'].nunique()} days | "
      f"median zero-day share {inter['zero_share'].median():.2f}")

# --- backtest all models --------------------------------------------------
s = cfg["forecast"]["sarima"]
models = {
    "seasonal_naive": lambda: SeasonalNaive(7),
    "moving_average": lambda: MovingAverage(28),
    "croston": lambda: Croston(),
    "sarima": lambda: SarimaPerItem(tuple(s["order"]), tuple(s["seasonal_order"]),
                                    min_obs=s["min_obs"]),
}
rows = []
for name, factory in models.items():
    res = rolling_origin(panel, factory,
                         horizon=cfg["forecast"]["horizon_days"],
                         n_splits=cfg["backtest"]["n_splits"],
                         step=cfg["backtest"]["step_days"])
    res.insert(0, "model", name)
    rows.append(res)
    print(f"{name:16s} mase={res['mase'].mean():.3f}  "
          f"peak_recall={res['peak_recall'].mean():.2f}")

results = pd.concat(rows, ignore_index=True)
run_dir = Path(cfg["paths"]["runs"]) / f"seed{cfg['seed']}"
run_dir.mkdir(parents=True, exist_ok=True)
results.to_csv(run_dir / "backtest.csv", index=False)
write_manifest(run_dir, cfg, extra={"n_items": int(panel["item_id"].nunique())})
print(f"\nwrote {run_dir}/backtest.csv + manifest.json")
