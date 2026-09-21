# Forecast-driven anticipatory re-slotting — data & forecasting layer

The ingest + forecasting half of the capstone. Everything here produces one
thing: a demand forecast with **regime labels** (peak / valley / normal) that
the dispatcher consumes to decide when to spend slack on re-slotting.

## The one design decision everything else follows from

`src/reslot/schemas.py` defines three canonical tables. Nothing downstream ever
sees a raw WMS column name. Consequences:

- The **synthetic generator and the real C&D extract are interchangeable.** The
  whole pipeline runs today, without the extract.
- When the extract arrives, the only file that changes is `io/demand.py`. If
  anything else has to change, the contract leaked and that's worth fixing.
- C&D data access is the project's single point of failure. This structure
  turns it from a prerequisite into a validation input.

## Layout

```
configs/default.yaml        every knob that changes a result
src/reslot/
  schemas.py                the contract — read this first
  io/onetrack.py            movement log loader (works, tested on real export)
  io/wms_picks.py           WMS pick report loader (yowmspd2_picks) -> EVENTS
  io/locations.py           location master loader -> SLOTS, reconcile()
  io/demand.py              order-line loader — STUB until the extract lands
  io/synthetic.py           parameterised demand generator
  features/panel.py         (item, date) panel, ABC, intermittency
  forecast/base.py          Forecaster interface — fit / predict
  forecast/baselines.py     seasonal naive, moving average, Croston
  forecast/sarima.py        SARIMAX per item, with fallback for sparse SKUs
  forecast/evaluate.py      rolling-origin backtest, accuracy + decision metrics
  travel/calibrate.py       travel-time model fitted from observed pick intervals
  utils/manifest.py         git SHA + config hash + seed on every run
scripts/
  ingest_onetrack.py        normalise the movement export
  validate_extract.py       run against the 5-row C&D sample FIRST
  run_forecast.py           panel -> backtest all models -> results + manifest
  calibrate_travel.py       fit travel model -> cost table for the sim + manifest
configs/travel.yaml         travel calibration knobs
docs/decisions.md           append-only decision log
```

## Run it

```bash
pip install -r requirements.txt
python scripts/ingest_onetrack.py            # against the real export
python scripts/run_forecast.py               # synthetic end-to-end
python scripts/calibrate_travel.py           # needs the two yowmspd2 CSVs in data/raw/
pytest -q
```

## Two things that are easy to get wrong

**1. Picks are not demand.** A pick timestamp is when the WMS released the
work, not when the customer ordered. Forecasting on picks means forecasting the
output of the policy we're trying to beat. `panel.from_pick_proxy()` exists for
pipeline debugging only and tags its output `source="pick_proxy"` so a
contaminated number can't reach the report unnoticed.

**2. Accuracy is not the objective.** The dispatcher needs the *timing* of
regime changes, not the level. `evaluate.decision_metrics()` measures peak and
valley recall within a tolerance, alongside MASE. A model can lose on MASE and
still win for this policy — reporting only MASE would understate the system.

## Travel model (current state)

C&D's WMS has no x/y/z; `WORK_PATH` does the job instead (see
`docs/decisions.md`). Fitted on 58,018 pick-to-pick legs, holdout split by trip:
fixed 26.8s per pick, 5.4s per case, 33.5s per aisle change, ~0 per path unit
within an aisle. Holdout MAE 62.7s vs 81.2s constant, trip-level Spearman 0.88.

**The number that shapes the project:** path travel plus aisle changes are
**8.8%** of observed case-pick time. That bounds what slotting can buy on the
pick face, and it points the policy at *fewer aisle changes per trip* rather than
shorter distance. Tagged `source="scan_interval_proxy"` — scan intervals are
not a time study. Reserve pallet pulls are untested and may look very different.

`data/processed/travel/travel_cost.csv` is the handoff to the simulator.

## Next

- `io/demand.py` RENAME map, once the analyst confirms column names
- confirm location-code semantics with the analyst (the parse now matches all
  1,543 real pick faces, but the meaning of each part is still inferred)
- run the travel decomposition on reserve pallet pulls
- ask what the `CR` / `LR` order-number suffixes mean
- calendar/promo exogenous regressors into SARIMAX
- handoff format from regime labels to the dispatcher
