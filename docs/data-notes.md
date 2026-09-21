# Data notes

## OneTrack movement export (received)

`1004WMS_ONETRACK_260910132159.xlsx` — 1,342 rows, 50 columns, ~24 minutes of a
single day (12:58–13:21). Sample only.

Event mix after normalisation: ship 458, putaway 213, pick 206, stage 186,
admin 125, count 82, production 42, internal_move 13, replen 12, adjust 5.
312 distinct items, 70 orders, 69 users.

**Rate:** ~1,342 rows / 23.6 min ≈ 57/min ≈ 80k/day ≈ tens of millions of rows
over 24 months. Ask for movement data scoped to a few representative months
(one peak, one trough) and restricted to pick / putaway / replen work types.
Demand at order-line grain is far smaller and can go full horizon.

**Open issue — date locale.** The filename stamp reads 2026-09-10 13:21:59;
the cells parse as 2026-10-09. Classic day-first / month-first collision.
`load_onetrack(expect_date=...)` warns on mismatch. Resolve with the analyst
before any timestamp-dependent result. Prefer ISO timestamps or a direct
database pull over an Excel export.

**Open issue — location codes.** Pick faces look like `KK038A`, `GG244A`,
`D161E`, `Q041B` → aisle letters + bay digits + level letter. `RT-xx` reserve,
`PL-xx` staging, `DK-xx` dock, `V-xxxx` vehicle. This is inferred, not
confirmed. Also ask whether a pick-path sequence number exists per location —
if there are no x/y/z coordinates, that sequence is the travel-cost proxy.
**Answered by the 9/14 location master — see below.**

Zones observed: `WZPCK` pick, `WZGNA`–`WZGNJ` general storage, `WZNDK`,
`WZPKG`, `WZEQP` equipment, `WYARD`, `WZSTG`, `WZDMG` damage, `WZRAW`, `WZVMS`.

## WMS pick report (received 9/14/26)

`yowmspd2_picks_past_30days_ranon9_14_26.csv` — 180,875 rows, 18 columns,
8/15–9/14/26. Every row TYPE `009` (pick). 1,114 items, 8,241 orders,
191 users. Loader: `io/wms_picks.py` → canonical `EVENTS`.

- **Two processes in one file.** Case picks from the pick face (`WZPCK`, UOM
  `CS`, ~66k rows, ~1,000 faces) and pallet pulls from reserve (`WZGNA`–`WZGNJ`,
  `WZPKG`, UOM `PL`, ~107k rows, ~27k locations). Forecast and model them
  separately.
- **Timestamps are unambiguous.** `MM/DD/YYYY hh:mm:ss AM/PM`, and the window
  includes days 15–31, so day-first misreads fail loudly. Parsed with an
  explicit format. The OneTrack locale issue does not apply to this export.
- **No eaches.** No `FROM_EACH`, so `qty_eaches` is null and
  `from_pick_proxy()` qty columns read 0. It now warns. Use `n_lines`:
  `intermittency(panel, value_col="n_lines")` gives median zero-day share 0.39.
- **`ORD_NO` is not numeric.** 1,399 rows (0.77%) carry suffixes —
  `10906764CR`, `10905011LR`. Probably credits/returns. **Ask the analyst**
  before counting them as outbound picks.
- **Labor Day (9/7)** ran 705 picks against ~6,000/day typical.

## WMS location master (received 9/14/26)

`yowmspd2_locations_9_14_26.csv` — 48,741 locations, 82 columns. Loader:
`io/locations.py` → canonical `SLOTS`.

- **Reconciles 100%** with the pick report: all 28,023 transacted locations
  exist in the master.
- **No coordinates.** `X`/`Y` on ~2% of rows (not racking), `Z`, `AISLE`,
  `RACK`, `WIDTH`, `DEPTH`, `STRG_CODE` empty. Matches what C&D said.
- **`WORK_PATH` is populated everywhere** and near-unique across pick faces.
  Pickers traverse in `WORK_PATH` order (median |Spearman| 0.93 over 3,118
  trips). This is the pick-path sequence asked about above, and it now carries
  the travel model. Ordinal, not metric.
- **Location-code parse confirmed on real data.** `parse_location()` matches
  all 1,543 pick faces. ~3,000 non-rack codes (`PRODGB401C`, `COOLER 14D`,
  numeric codes) don't match and return null, as intended. Semantics still
  worth confirming with the analyst.
- `TYPE`: `F` pick face (1,754), `R` reserve (46,833). 1,543 `F` in `WZPCK`, of
  which 541 saw no pick in the window — the free slack.
- Also populated and potentially useful: `PTWY_ZONE`/`PTWY_PATH` (putaway
  routing), `LAST_PICK`, `LAST_PUT`, `TRANS_CNT`, `STD_CPCT` (1 on pick faces).

## Order-line extract (requested, not received)

Target schema: `schemas.DEMAND_LINES`. What defines the ask:

- **Grain** — one row per order line, never pre-aggregated
- **Timing** — order received datetime, separate from requested ship and actual ship
- **Quantity unbundled** — ordered / allocated / shipped / short / cancelled
- **History** — 24 months, two seasonal cycles plus a holdout

Run `scripts/validate_extract.py` against the 5-row sample before the full pull.
`quality_report()` checks each answer we were given: are cancels retained, is
order_received_ts genuinely distinct from ship date, how many months landed.
