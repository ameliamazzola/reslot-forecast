# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Data + forecasting layer of a capstone on forecast-driven anticipatory re-slotting for the C&D York DC. It ingests WMS extracts, forecasts demand with peak/valley/normal regime labels, and builds a travel model (seconds from pick scans, feet from the layout drawing) for a downstream simulator/dispatcher that does not exist in this repo yet.

## Commands

Windows, Python 3.14 venv at `.venv/`. Run everything from the repo root: config and data paths are relative to the CWD.

```bash
pip install -r requirements.txt
pytest -q                                              # all tests
pytest tests/test_travel.py::test_split_no_trip_leak -q  # one test
python scripts/run_forecast.py                         # synthetic end-to-end forecast + backtest
python scripts/ingest_onetrack.py                      # normalise the OneTrack movement export
python scripts/validate_extract.py                     # run against a C&D sample before trusting it
python scripts/calibrate_travel.py [configs/travel.yaml]       # needs yowmspd2 CSVs in data/raw/
python scripts/build_travel_network.py [configs/travel.yaml]   # also needs the DXF; reads ~240 MB, takes minutes
```

No packaging/install of `reslot`: scripts and tests do `sys.path.insert(0, <repo>/src)`. Follow that pattern in new scripts/tests. No linter is configured.

Tests marked `needs_outputs` in `tests/test_travel.py` are skipped unless `data/processed/travel/` has been generated from real data; they assert hand-traced distances on the drawing.

## Architecture

**The schema contract is the core design.** `src/reslot/schemas.py` defines canonical tables (`EVENTS`, `DEMAND_LINES`, `ITEM_MASTER`, `PANEL`, `SLOTS`, `TRIP_LEGS`) and `validate(df, schema, name, strict)`. Loaders in `src/reslot/io/` are the only code that sees raw WMS column names (each has a `RENAME` map). Synthetic data (`io/synthetic.py`) and the real extract are interchangeable; when the real order-line extract arrives only `io/demand.py` (currently a stub) should change. If anything else has to change, the contract leaked.

**Config.** `reslot.config.load_config(path)` returns a plain dict from YAML and refuses a config without `seed`. Every result-changing knob lives in `configs/` (`default.yaml` for forecasting, `travel.yaml` for calibration/network/crosswalk/staging), never in code.

**Manifests.** Every script ends with `utils.manifest.write_manifest(runs/<name>/, cfg, inputs, extra)`: git SHA, dirty flag, config hash, seed, input file hashes, metrics. `git_dirty: true` means the run is not citable. New run scripts should do the same.

**Forecast path.** `features/panel.py` builds the `(item, date)` panel → `forecast/` models implement `Forecaster.fit/predict` (`base.py`) → `forecast/evaluate.py` does rolling-origin backtests reporting both MASE and `decision_metrics` (peak/valley recall within a tolerance). Regime timing, not accuracy, is the objective.

**Travel has two layers that are not yet connected:**
- `travel/calibrate.py` — `TravelModel`: `seconds = fixed + per_case*qty + cross_aisle*I + isotonic(delta_path)`, fitted by median regression on consecutive pick scans. `delta_path` is in WMS `WORK_PATH` units (ordinal, not feet). `move_cost()` is repositioning cost (no per-case term). Output handed to the sim is `data/processed/travel/travel_cost.csv`; coefficients are in `runs/travel_seed<seed>/manifest.json` (the isotonic curve itself is not persisted).
- `travel/network.py` + `crosswalk.py` + `lanes.py` — DXF layers → networkx graph → `stop_distance_ft.csv` (feet), `location_stop_crosswalk.csv` (WMS location → `stop_id` by rule), PL staging lanes → lane stops. Stop IDs are coordinate-derived (`S_/J_/L_/D_/LN_<x>_<y>`) and are the join key across outputs. Trips start at the single `D_` dispatch stop; drop-off lanes are provisional (`drop_off_mode`).
- There is no feet→seconds conversion yet; the time model has not been refit on network feet.

`src/affinity/` (fpgrowth pairs, Louvain SKU clusters, join with the pick-count forecast) and `src/forecast/Pick_Count_Forecast.ipynb` are standalone scripts/notebooks with their own hardcoded paths, not part of the `reslot` package. `notebooks/reslot_algorithm.ipynb` is an exploratory golden-zone swap evaluated in feet.

## Rules that are easy to break

- **Picks are not demand.** Pick timestamps are WMS release times, i.e. the output of the policy being beaten. `panel.from_pick_proxy()` is for debugging and tags `source="pick_proxy"`; anything fitted on scan intervals carries `source="scan_interval_proxy"` plus a caveat. Keep these tags flowing into outputs.
- **Travel holdout splits by trip, never by leg** (`split_by_trip`, guarded by `test_split_no_trip_leak`). Several tests guard design decisions; read the docstring before "fixing" a failing one.
- **Quantity stays in the travel model.** Dropping it attributes handling time to distance and inflates the payoff of re-slotting. Path travel + aisle changes are only ~8.8% of case-pick time; the policy lever is fewer aisle changes per trip.
- **Slot inventory comes from the location master (`SLOTS`), not from transactions** — untouched faces are the free capacity re-slotting moves into. PY/PZ (tall/short pallet slot) is a hard re-slotting constraint.
- **Drawing semantics:** lines connect only at shared endpoints or stop circles; crossings are never auto-joined. Misfiled entities are reported, never guessed at.
- Timestamps in WMS extracts are parsed with an explicit format (`%m/%d/%Y %I:%M:%S %p`); never let pandas infer them.
- Read WMS ID columns as strings (order numbers carry suffixes like `CR`/`LR`; `TYPE` is zero-padded).

## Data handling

Raw C&D data, everything in `data/interim/` and `data/processed/`, `runs/`, and layout drawings (`*.dxf`, `*.dwg`) are gitignored and must never be committed — permissions with C&D are not settled. Teammates regenerate derived outputs from the shared Drive extracts.

## Docs

- `docs/decisions.md` — append-only decision log (decision / why / reverses if). Add an entry for any modelling choice.
- `docs/data-notes.md` — facts about the extracts and the drawing.

# RackSmart project context
@Background.md
@decisions.md

## Rules for this repo
- Reuse travel/network.py and travel/crosswalk.py for all travel; never write new distance math
- Never read or open raw C&D extracts; use data/sample/ only
- Nothing in data/processed/ is committed
- Every run writes a manifest: config hash, git SHA, seed, date window, operators, pd/drop-off mode
- Propose a plan before editing; changes as reviewable diffs on a branch
- Static = no re-slots; one_shot pays for its moves (see decisions.md)