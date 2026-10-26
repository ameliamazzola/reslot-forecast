# Decisions

Append-only. Each entry: what was decided, why, and what would reverse it.
Data facts live in `data-notes.md`; this file records choices.

---

## 2026-09-14 — WORK_PATH is the travel coordinate

**Decision.** Use the WMS `WORK_PATH` field as a one-dimensional travel
coordinate rather than waiting on x/y/z or CAD-derived geometry.

**Why.** C&D confirmed the WMS does not populate coordinates, and the location
master bears that out. `WORK_PATH` is populated on every row and near-unique
across pick faces. Across 3,118 order-trips of 5+ picks, median |Spearman|
between chronological pick order and `WORK_PATH` is 0.928; 66% of trips exceed
0.8. Pickers traverse in path order.

**Reverses if.** The CAD layout gives metric coordinates, or the analyst says
`WORK_PATH` is maintained per zone rather than building-wide. `WORK_PATH` is
ordinal — it orders slots, it doesn't measure feet. Metric geometry is still
worth having.

---

## 2026-09-14 — Travel model includes quantity and aisle changes

**Decision.** Model pick-to-pick time as
`fixed + per_case*qty + cross_aisle + distance(delta_path)`, not distance alone.

**Why.** Cases picked predicts elapsed time better than path distance does
(Spearman 0.52 vs 0.32). Holdout MAE: constant 81.2s, distance-only 76.0s,
full model 62.7s. Omitting quantity attributes handling time to distance and
inflates the apparent payoff of re-slotting.

**Consequence.** Path travel plus aisle changes are 8.8% of observed case-pick
time. An aisle change costs ~33s; in-aisle distance is nearly free. The policy
lever on the pick face is reducing aisle changes per trip (co-locating SKUs that
ship together), not shortening distance to a golden zone. And 8.8% is the
ceiling any slotting policy can address there.

**Reverses if.** Reserve pallet pulls decompose differently — plausible and
untested. Run the same decomposition there before settling where the policy
aims.

---

## 2026-09-14 — No Chebyshev travel model for C&D data

**Decision.** Chebyshev (simultaneous-axis AS/RS crane) is not used for the C&D
operation.

**Why.** This DC is operators on equipment moving through an aisle network,
not a crane. Travel runs along aisles.

**Reverses if.** A synthetic AS/RS becomes the primary subject again.

---

## 2026-09-14 — Slot inventory comes from the master, not transactions

**Decision.** Build the slot model from `SLOTS` (location master).

**Why.** 1,543 pick faces exist in `WZPCK`; 1,002 saw a case pick in 30 days.
The other 541 are the free capacity re-slotting moves into, and they appear
only in the master.

---

## 2026-09-14 — Trip-level holdout split

**Decision.** Split travel-model train/holdout by trip, never by leg.

**Why.** Legs in one trip share an operator, an hour and an aisle. A leg-level
split leaks correlated observations across the split and flatters the error.
Guarded by `test_split_no_trip_leak`.

---

## 2026-09-14 — Derived travel outputs are not committed

**Decision.** `travel_cost.csv` and `travel_profile.csv` go to
`data/processed/travel/` (gitignored). Each teammate regenerates them from the
Drive extracts.

**Why.** They're derived from C&D data and permissions aren't settled. Same
rule as the rest of `data/processed/`.

**Reverses if.** C&D confirms aggregates are fine to commit. Then commit the
cost table as the fixed sim↔data interface.

---

## 2026-09-28 — Metric travel network from the layout drawing

**Decision.** Build travel geometry from the York DC layout drawing: forklift
paths and stop points drawn on dedicated layers, turned into a graph and a
stop-to-stop distance table in feet (`travel/network.py`). This is the
"reverses if" of the WORK_PATH entry above, for geometry: WORK_PATH stays the
calibration history until the fit is rerun on network feet.

**Why.** WORK_PATH is ordinal. It can't say how far two slots are apart, so it
can't price a re-slot or tell a golden zone from a far one.

**How it's built.** Lines connect only where they share an endpoint or a stop
circle; crossings are not auto-joined, so every turn is drawn explicitly.
Hand-traced routes match the table (placeholder→FF bays 1–4 133.7 ft, →FF
charger end 705.3 ft, FF→GG across either cross-aisle 41.1 ft).

**Reverses if.** A site walk finds a drawn path that trucks can't use, or an
opening that isn't drawn.

---

## 2026-09-28 — Location → stop rule (crosswalk)

**Decision.** Map each WMS location to a drawn stop by rule
(`travel/crosswalk.py`, anchors in `configs/travel.yaml`): aisle code → path x
(FF rightmost), bay 1 at the dock, 4 bays per stop (one pallet per bay, odd
east / even west), each third of the aisle anchored to its own drawn y.
Tunnel bays 79–84 and 185–190 (upper levels only) map to the cross-aisle
junction under them. Level doesn't change the stop.

**Why.** No coordinates in the WMS. Evidence the rule holds: all 30 drawn
column rack sections land exactly on bay numbers the WMS skips; no stop gets
more than 4 floor locations; floor spot-check by Anthony agreed.

**Scope.** Case picking on FF–QQ. 97.1% of CS picks in the 30-day report map to
a stop (64,763 / 66,699); the rest are RR, ZZ, LP, W — not drawn. Reserve
pallet pulls (single-letter aisles, double-deep) are out of scope.

**Reverses if.** A spot-check disagrees, or the analyst's location-code
definitions contradict the side/level reading.

---

## 2026-09-28 — Trip start/drop-off is a placeholder

**Decision.** One placeholder node (`PD_Placeholder` layer) on a 35 ft spur
off the top cross-aisle, between LL and KK, until staging-destination data
arrives. `network.pd_mode: placeholder` is written to every manifest.

**Why.** `TO_LOCATION` on case picks is the picker's equipment (V-xxxx), not a
floor location, so the transactions can't say where pallets go.

**Consequence.** Pick-to-pick legs are unaffected. Trip totals that include the
first/last leg are not citable until `pd_mode: real`. Run a sensitivity check
across candidate drop-off points before reporting any policy ranking.

---

## 2026-10-18 — Travel network covers the whole building

**Decision.** Extend the network and crosswalk from FF–QQ to every racked aisle:
EE, DD, CC, BB, AA and Z–A (no I/O), 29 new paths east of FF. Same location →
stop rule and the same y anchors as FF–QQ; only `aisle_x` was added.

**Why the aisle order is right.** Letters continue east of FF, and the drawn
gaps line up with the WMS bay structure aisle by aisle: J, K, L have no bays
79–214 and their paths have no stops there; M, N, P, Q have no bays 79–148
and neither do their paths; X and Y stop at bay 184 and their paths are short.
Every stop still gets at most 4 floor locations, and all 30 column sections
still land on skipped bays.

**Result.** 2,229 rack stops, 184 junctions, one connected graph. 42,462 rack
locations map to a stop (was the FF–QQ set only). Reserve pallet pulls: 98.0%
of rack-coded PL picks map; the other ~16.7k PL picks are off the production
lines (WZPKG), which aren't racks. CS pick coverage unchanged at 97.2%.

**Builder changes.** (1) Circles drawn twice at the same point are one stop
(108 merged). (2) Pieces of path with no stop are dropped and listed (two
1-inch strays) — they can't change a route; a disconnected piece WITH a stop
still fails the build. (3) Circles on path layers and lines on stop layers are
ignored and listed, never guessed at. Crossings are still not auto-joined.

**Drawing revisions (Oct 18, second pass).** P bottom stop added, CC tunnel
junctions moved to `Stop_Junction`, missing junction circles added. The
placeholder moved to (180174.4, 6094.1), between JJ and HH, on a 32 ft spur
off the top cross-aisle (was between LL and KK); its hand-calc routes are now
100.3 ft to FF bays 1–4 and 671.9 ft to the FF charger end.

**Still open.** (1) The placeholder must sit on `PD_Placeholder` with a drawn
spur line and a junction circle where the spur meets the top cross-aisle.
(2) X/Y bays 157–184: the site says no racks below the y=1544 cross-aisle,
but the WMS has 304 locations there, 274 occupied, with 771 picks Sep 15–Oct
15. Until it's known where those pallets physically sit, they stay
`check_snap`. (3) DD vs CC is assigned by order, not yet checked on the floor.

**Reverses if.** A floor check finds a letter on the wrong path, or the site
says the east aisles are numbered from the other end.

---

## 2026-10-19 — Trip ends: dispatch board and staging lanes

**Decision.** Trips start at `Stop_Dispatch` (the dispatch board by the office,
where operators get jobs) and drop off at `Stop_Lane` circles (100 outbound
lane mouths, cross-aisle end, 25 groups of 4). `PD_Placeholder` is retired.

**PL codes to lanes (provisional).** The WMS has PL01–PL57 (zone WZNDK); the
drawing has 100 lanes. Rules agreed with Anthony: PL01 is the rightmost group
and numbers ascend west; 44-pallet PLs get 2 lanes, 66/88 get 3 (131 lanes
asked); one lane may serve two neighbouring PLs (31 do). Placement minimises
PLs split across a group gap (5: PL23, 38, 46, 51, 54). PL..A codes are inbound
receiving staging (they carry ~30k of the month's inbound putaways) and PL60
has no lane; both are out of scope for now. A 2-lane PL routes to its right
lane, a 3-lane PL to its middle lane.

**Hand-calcs.** Dispatch → FF bays 1–4: 224.2 ft. Dispatch → lane 1 (PL01):
456.3 ft.

**Reverses if.** The site gives a lane-to-door map; then it replaces the
convention and `drop_off_mode` flips to `real`.

---

## 2026-10-19 — Production lines are stops

**Decision.** `Stop_Line` circles mark where a forklift takes a finished pallet
off each production line; a TEXT on `Line_Label` next to each names it with
the WMS code. L1–L5 have one pickup each, L6 and L7 two; a code with two
pickups maps to the one nearer the dispatch board. "Empty Pallets" and
"Slip sheets" are drawn supply points with no WMS location yet. D01 (534
moves) was not found on the drawing and stays unmapped.

**Supervisor's location key (Oct 19) agrees with the model.** Bays start at
the dock and run toward the chargers; putaway-zone section letters A/B/C line
up with the crosswalk's top/middle/bottom segments (10,317 / 14,242 / 9,356).
PY = tall-pallet slot, PZ = short; HH = household (A–Y), personal care Z–QQ,
GG/HH partial pallets. PY/PZ is a hard constraint for re-slotting.

**Hand-calc.** Dispatch → L3: 436.9 ft (down Y, along the upper cross-aisle).

## Affinity clustering + forecast join (Oct 2026)

**Data filters (build_matrix.py)**
- VMS: drop SKUs picked from zone WZVMS (discontinued Vitafusion/L'il Critters finished goods,
  ingredients, packaging), except active brands stored there (VMS_KEEP_PATTERN = "ARRID").
- Discontinued vitamin lines dropped wherever stored (VITAMIN_PATTERN = "VITFUS|CRITTER").
  Viviscal is active and is kept.
- Full-pallet picks included.

**Affinity settings**
- Transaction = customer order; item = SKU.
- fpgrowth: MIN_SUPPORT = 0.005, MAX_LEN = 2.
- Edge weight = Jaccard; strong edges MIN_JACCARD = 0.10.
- Louvain: resolution = 1.0, seed = 42.

**Results**
- 8 clusters, modularity ≈ 0.52.
- Robustness: ARI 0.81–1.00 vs. baseline across Jaccard 0.10–0.20 at resolution ≤ 1.0.
- Two disconnected components: personal care (218 SKUs) and household (129), plus Platinum litter (3).
- Clustered SKUs = 26% of SKUs, 66% of pick lines.
- Household clusters mostly full-pallet → place by dock distance; personal care mostly case-pick → place by adjacency.

**Forecast join (combine_with_forecast.py)**
- Affinity measured by order co-occurrence; demand measured in pick lines.
- Anticipatory candidate rule: change_flag = "up" AND demand_group in {Steady, Variable}.
- New SKUs (< 28 days history) have no forecast; need a placement rule.   