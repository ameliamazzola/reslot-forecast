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

Zones observed: `WZPCK` pick, `WZGNA`–`WZGNJ` general storage, `WZNDK`,
`WZPKG`, `WZEQP` equipment, `WYARD`, `WZSTG`, `WZDMG` damage, `WZRAW`, `WZVMS`.

## Order-line extract (requested, not received)

Target schema: `schemas.DEMAND_LINES`. What defines the ask:

- **Grain** — one row per order line, never pre-aggregated
- **Timing** — order received datetime, separate from requested ship and actual ship
- **Quantity unbundled** — ordered / allocated / shipped / short / cancelled
- **History** — 24 months, two seasonal cycles plus a holdout

Run `scripts/validate_extract.py` against the 5-row sample before the full pull.
`quality_report()` checks each answer we were given: are cancels retained, is
order_received_ts genuinely distinct from ship date, how many months landed.
