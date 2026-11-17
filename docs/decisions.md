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

2026-11-02 — Static policy means no re-slots
Decision. "Static" runs the week on the location master's slotting as-is, with no moves during the run. The notebook's one-time start-of-week swap is a separate policy (one_shot), not static.

Why. Static is current practice: the layout the site actually ran. It is the reference every other policy is compared against, alongside reactive ABC as the honest re-slotting baseline. Folding a swap into static would mix current practice with a policy and blur that reference.

Consequence. one_shot is charged the forklift time for its relocation moves, the same as reactive ABC pays for its moves. A policy that re-slots for free is not a fair comparison. The static replay is not a sim validation; see the out-of-time entry below.

Reverses if. The team agrees a different "current practice" definition (e.g. the site's own periodic re-slot cadence) once C&D describes it.

2026-11-02 — First sim run uses the last full week of the 30-day extract
Decision. Mon Oct 5 – Sun Oct 11, 2026. Start and end dates live in the YAML config and are written to every manifest, never hard-coded.

Why. Most recent full week, so it is the week most likely to match the location master snapshot (the master shows where things sat when exported, not historically). A full Mon–Sun week captures the weekly cycle. It also falls after the Aug 15–Sep 14 calibration window, so it serves as the out-of-time check of the leg model.

Consequence. Picks that don't map to an ok stop are dropped and counted by reason per run (CS coverage ~97.2%; X/Y bays 157–184 still check_snap). Results for this week are an out-of-time check, not a policy finding; policy rankings need multiple weeks.

Reverses if. The location master snapshot date turns out to be far from Oct 5–11, or the site flags that week as abnormal (outage, inventory count, unusual staffing).

2026-11-02 — Operator count is a config value
Decision. operators is a fixed number in the YAML config, not read from the data at runtime. The first runs use the observed count for the week as a reference value: 6 operators, the median over clock hours with at least one case pick (WZPCK/CS) in Oct 5–11 of distinct USER_NAME per clock hour (transactions_0915_1015.csv; p90 9, max 12). This counts distinct operators per clock hour and is not a measured headcount: an operator who picks on both sides of an hour boundary counts in both hours. It agrees with an example shift from the site (5 case pickers). Whole-week distinct case pickers (91) is not used, because it counts operators who never overlapped. The shift log's 30 is not used: it covers all roles on a shift, not case picking. For later reference: pallet picks (PL, excluding production lines WZPKG) median 13 per clock hour (p90 17), against 8 full-pallet pickers on the example shift; that gap is unexplained. Case and pallet combined median 19 (p90 24), with 150 distinct operators over the week.

Why. Reading it from the data at runtime would tie the sim to one week and remove the what-if lever the October feedback asked for (changeable operator count). Starting from the observed count keeps the first runs anchored to the real operation.

Consequence. Under static, leg times don't depend on operator count, and queueing is understated while dispatch legs are untimed (see the dispatch-legs entry). Operator-count results are therefore not findings yet. One number for the whole week is coarse; staffing differs by shift.

Reverses if. Once dispatch legs are timed, sweeps show results are sensitive to shift-to-shift staffing; then replace the single value with a per-shift schedule in config.

2026-11-02 — Leg time refit on network feet (candidate)
Decision. Refit the 2026-09-14 model with distance = feet(previous stop, this stop) from stop_distance_ft.csv: same form (fixed + per_case×qty + cross_aisle×I + isotonic distance), same median regression, and the same held-out trips as the WORK_PATH fit (rebuilt from the picks; leg and trip counts checked against runs/travel_seed20260914). Legs touching a pick with no ok stop are dropped, never merged. Adopt for the sim only after the holdout review below.

Review rule (fixed before results). Both models are scored on the identical set of held-out legs: the WORK_PATH model is re-scored on the feet model's leg subset, not quoted at 62.7 s. For each model, report holdout MAE and improvement over its own constant baseline. Bootstrap the MAE difference (feet − WORK_PATH) by resampling held-out trips, 1,000 resamples, seed in config. The feet model is rejected only if the 95% interval lies entirely above zero; otherwise it is adopted.

Why. 2026-09-28: WORK_PATH is ordinal and can't price a re-slot. The network is the geometry, and CLAUDE.md routes all travel through it.

Consequence. Legs keep source="scan_interval_proxy". The 8.8% addressable share is recomputed on feet and may move. Legs touching RR, ZZ or X/Y check_snap picks (~3%) drop out of both fit and holdout.

Reverses if. The bootstrap interval shows the feet model worse than WORK_PATH on identical legs, which would point at the drawing or the crosswalk. Outcome (Adopted / Rejected, with MAEs and interval) is appended to this entry after the review.

2026-11-02 — Dispatch legs untimed; case-pick trips return to dispatch (provisional)
Decision. Tours start and end at Stop_Dispatch. Dispatch→first pick and last pick→dispatch are counted in feet only. An operator is busy from first to last pick scan. Every sim manifest records dispatch_legs: untimed and drop_off_mode: provisional.

Why. The leg model is fitted on pick-to-pick legs only, and there's no measured forklift drive speed for long runs. The 2026-10-19 PL→lane map can't be applied: TO_LOCATION on case picks is the truck (2026-09-28), outbound DROPOFF rows record no PL (data-notes), and the 365-day export has no TO_LOCATION.

Consequence. Operator busy time and queueing are understated. Queueing and operator-count results are not findings until dispatch legs are timed. Tour totals that include drop-off are not citable.

Reverses if. (1) A dispatch-leg time is estimated from the data: the gap between an operator's last pick on one tour and first pick on the next, fitted against network feet for the dispatch/drop-off round trip. (2) Drop-off data arrives (TO_LOCATION on drop-off rows, or an order→PL/door map). (3) C&D's equipment motion parameters give a drive speed to cross-check (1).

2026-11-02 — The static run is an out-of-time check of the leg model, not a sim validation
Decision. Primary metric: simulated vs observed pick-to-pick leg seconds for Oct 5–11, first/last legs excluded, with the calibration's leg filters. Secondary: first-to-last scan span, measured the same way on both sides. min_picks applies to these statistics only; every tour is simulated. The model checked is whichever leg model is adopted under the 2026-11-02 refit entry.

Why. Under static, a simulated leg is the model's prediction for the observed leg and queueing doesn't touch it, so this tests whether the Aug 15–Sep 14 fit holds in October, not the simulation.

Source check. If the fit and the Oct 5–11 legs come from different exports (365-day vs 30-day), first confirm both exports produce the same legs on overlapping dates (scan interval proxy, filters, tour definition). A mismatch is an export difference, not model drift, and is fixed before the check is reported.

Consequence. Sim mechanics (queueing, operator count) have no validation yet and are not reported as findings.

Reverses if. An observed measure of operator busy time or queueing becomes available to compare against.

2026-11-03 — Sim window extended to everything after the planning month
Decision. Replay Sep 21 – Oct 15, 2026 (25 days; Oct 15 may be partial), with each method's plan applied on Sep 21. Replaces the Oct 5–11 window of the 2026-11-02 entry. Dates live in configs/sim.yaml and every manifest.
Why. One week set the full one-time move cost against one week of savings, so every plan looked like a loss even when it paid back in under two weeks. Replaying all the post-planning data the transactions export covers gives a simulated net over 3.5 weeks instead of a projection, and is closer to the notebook's monthly evaluation.
Consequence. Still one plan per method, applied once; no re-planning during the run. The leg-model check covers the same 25 days, still after the Aug 15–Sep 14 fit, so it remains out of time. The affinity clusters overlap the run by one day (Sep 21). The operator reference value (6) was measured on Oct 5–11.
Reverses if. A later transactions export allows a full month after the planning window, or the team moves to monthly re-planning as in the notebook.

2026-11-16 — Planning inputs end at the run start; named exceptions only
Decision. The notebook methods plan on: (1) every face the structure allows (crosswalk ok, WZPCK/F, stop in the distance table), not the faces picked in any window; (2) the item at each face = the last one picked there before Sep 21; (3) affinity clusters built from picks before Sep 21 (cutoff recorded in sku_clusters_meta.json); (4) case counts from the Oct 20 on-hand snapshot. Item (4) is the only input allowed past the run start; it is named in lookahead_exceptions, and the engine refuses any other planning input that ends after the start.

Why. Two leaks let the plan see the run it was scored on: (a) the candidate face set included faces picked during the run window; (b) planning reads ran one day past the planning end, so tours that crossed midnight into Sep 21 carried run-day picks into the plan. Found by the planning-invariance test (commit 4943bcd); fixed in 0358049, where restoring the old read window makes the test fail in all 6 cases.

Consequence. Results from before 0358049 are not citable. In practice: they were likely optimistic, because the plan had hints about the run it was scored on. Any place they were shown (progress slides, report drafts, sponsor updates) gets a note that they were superseded, with the corrected figures. Case counts are tested with a snapshot-free sensitivity run: cases per pallet × 0.75 face fill. The 0.75 is an assumption, not a measurement. Before/after per policy: [fill in from compare_runs].

Reverses if. An on-hand snapshot from before Sep 21 is found; then move_cases stops being an exception.

2026-11-16 — Both sim windows are reported; the window is now fixed
Decision. Results are reported for both windows: Oct 5–11 (one week, 2026-11-02) and Sep 21–Oct 15 (everything after the planning month, 2026-11-03). No further window changes after seeing results.

Why. The longer window was adopted after the one-week result showed a net loss. The reason it is still appropriate: re-slotting moves are paid once, while savings accrue every day, so one week is too short to recover move costs. Reporting both makes the dependence on window length visible instead of hiding it.

Consequence. Any headline figure states its window. If a policy is negative on one week and positive over 25 days, that is reported as a payback-period finding.

Reverses if. Newer pick data allows a longer out-of-sample window; it is then added as a third window, not swapped in.