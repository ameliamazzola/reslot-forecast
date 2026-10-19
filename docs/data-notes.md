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

## York DC layout drawing (received from site engineering)

AutoCAD drawing, exported to DXF for processing (`DC_Layout_Paths_5.dxf`,
~240 MB). Lives in `data/raw/` only — it is C&D's blueprint.

- **Units and frame.** Inches, AutoCAD World coordinates. AutoCAD may display
  feet-inches (Architectural units); the stored values are inches.
- **Orientation.** Dock at the top (high y), chargers at the bottom. Pick aisles
  run vertically (along y). FF is the rightmost aisle at x = 180855.63; aisles
  are 236" apart, rack sections 103".
- **Layers added by us.** `Path_Aisle`, `Path_Cross` (lines), `Stop_Rack`,
  `Stop_Junction`, `PD_Placeholder` (3" circles). Exported names carry a leading
  apostrophe; the loader strips it.
- **Cross-aisles are tunnels.** Top (y 5710), upper third (y 3527, the bowtie
  sections), middle (y 798), bottom (y −1410). Bays 79–84 and 185–190 exist only
  at levels D/E, over the upper and middle cross-aisles.
- **Columns.** Layer `-ST_R_SD_COL`; 30 rack sections with a building column,
  every ~612". The WMS skips one bay number at each (e.g. GG odd 17, 41, 65…).
- **Location code, now confirmed with the site.** `GG184A` = aisle `GG`, bay
  `184`, level `A`. One pallet per bay. Odd bays east face, even west. Level A
  is floor (pick face), B–F reserve above. Bay 1 at the dock end.
- **Zones on the racks.** Household aisles A–Y, personal care Z–QQ, GG/HH for
  partial pallets. Putaway zone codes like `PZHHB`: `PZ` short pallet / `PY`
  tall, `HH` household, last letter front/middle/back section. These are
  re-slotting constraints (product family, pallet height).
- **Full-building paths (Oct 18).** 29 aisle paths added east of FF, x from
  181208.63 (EE) to 190784.63 (A). Aisle spacing alternates 336"/348" in the
  double-deep block. Same rack-row y values as FF–QQ (top stop 5571.4, half
  section 3634.9, middle start 3398.4, bottom start 669.4); cross-aisles at
  5710.4, 3526.9, 797.9 (gap between M and H), and a bottom cross at −1287.6
  (−1081.6 from F to A). Extra cross pieces at 1544.4 (under X/Y), 1853.4 and
  154.4. DD and CC are zig-zag paths (jogs at y 154–360).
- **Drawing fixes (status after the Oct 18 second pass).**
  - Done: P bottom stop (186344.63, −1184.59); CC junctions at
    (181892.6, 3526.9) and (181892.6, 797.9) on `Stop_Junction`; junction
    circles at (181592.6, −1410.1) and (181856.6, −1410.1); stray stubs
    across G and F at y ≈ 3604 removed.
  - Open: the placeholder circle at (180174.4, 6094.1) is on the `Pallet`
    layer, which also holds a 4 × 11 grid of 44 circles in the staging lanes
    (x 188230–188380). Move the one circle to `PD_Placeholder`, draw a line
    straight down to (180174.4, 5710.41), and put a `Stop_Junction` circle
    there.
  - Open: X/Y bays 157–184 — site says no racks below the y=1544
    cross-aisle, yet the WMS has 304 locations (274 occupied, zone WZGNH) and
    771 picks Sep 15–Oct 15 there. Need to know where those pallets sit
    (floor-stacked? numbered elsewhere?).
  - Harmless, handled by the builder: 108 duplicate circles, two 1-inch
    stubs near (181559.6, −1239) and (181889.6, −1239), 24 duplicate circles
    on `Path_Cross` at y −1287.6, 2 duplicate DD/CC aisle lines on
    `Stop_Junction`.
- **Staging and dispatch (Oct 19 drawing).** 100 `Stop_Lane` circles at
  y = 5766.6 (56" spurs to the top cross-aisle), x 188380.6 (lane 1, east) to
  178812.9 (lane 100). Lanes 50" apart in groups of 4; groups 192–225" apart;
  1,234" office gap between lane 52 and 53. `Stop_Dispatch` at
  (183183.9, 5933.3) on a 223" spur. 1,100 circles on `Pallet` are pallet
  positions (reminder only, not read). A leftover `Dock_aisle` line at
  x 180174.4 (the old placeholder spur) is not read; delete it or say what
  it is.
- **PL locations in the master.** PL01–PL57 + PL60 in WZNDK (capacity 44 / 66
  / 88), PL08A, 13A, 17A, 25A, 35A, 38A, 47A, 50A also WZNDK (inbound), PL101,
  PL101A/B, PL102A elsewhere. Outbound DROPOFF rows don't record a PL yet --
  need TO_LOCATION on the export to know which lane a pallet went to.
- **Production lines (Oct 19 drawing).** 11 `Stop_Line` circles labelled on
  `Line_Label`: L1 (188089.5, 3285.9), L2 (187544.3, 3285.9), L3 (185783.2,
  3290.3), L4 (186544.3, 3285.9), L5 (186902.0, 3283.5), L6 ×2 (188063.1,
  328.8 / 399.5), L7 ×2 (187407.4, 327.8 / 398.9), Empty Pallets (186903.0,
  2285.1), Slip sheets (186824.3, 2510.4). New connector at x 186747.6 meets
  the y=1853.4 cross piece without a junction circle -- add one.
- **Not drawn.** RR, ZZ (multi-level case pick), non-aisle location families
  (BW, GB, LS, VA–VN), production lines, staging lanes (along the top,
  y ≈ 5830–6310).
