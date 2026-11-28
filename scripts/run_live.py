"""Live mode: generated orders on a real-time clock, one policy, served on 127.0.0.1.

Run:  python scripts/run_live.py [configs/sim.yaml] [--policy static|racksmart|abc_cost_test|family_grouping]
                                 [--seed N] [--speed 1|10|60|600] [--port 8765] [--hours H]
                                 [--sku-to-face real|synthetic]

  source   GeneratorSource (config generator; empirical mode by default), profile
           from generator.profile_path, seed from config or --seed. Case picks only.
           Layout --sku-to-face (default real: the last item picked at each face
           before the run start). Each tour's picks in visit order by
           generator.routing (calibrated: scripts/calibrate_routing.py).
  policy   static, or a notebook method planned from the planning month exactly as
           replay plans it (racksmart.load_racksmart), with the no-lookahead check
           against the run start (engine.check_planning_inputs).
  clock    simpy.rt.RealtimeEnvironment(strict=False) at live.speed sim seconds per
           wall second, changeable while running (POST /control); starts at the
           generator window start (config window.start).
  server   standard library only, bound to 127.0.0.1 (anything else is refused):
             GET  /         the ui/ folder, as static files
             GET  /events   Server-Sent Events: one snapshot at most
                            live.max_events_per_second times a second (wall clock)
             GET  /state    the latest snapshot as JSON
             POST /control  {"speed": one of live.speeds} and/or {"affinity": 0..1};
                            both go into the source's control log with their sim time

Runs until the window ends (or --hours of sim time), or Ctrl+C. Then writes
<paths.out>/live_<timestamp>/manifest.json: mode live, citable false, the source
(with its control log), seed, policy, speeds used, wall-clock start/end, sim time
covered, and totals. Never citable.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from reslot.config import load_config
from reslot.sim import engine as E
from reslot.sim import generator as G
from reslot.sim import live as L
from reslot.sim.policy import make_policy
from reslot.sim.racksmart import METHODS
from reslot.travel.artifacts import load_travel
from reslot.utils.manifest import write_manifest

ap = argparse.ArgumentParser(description="Live mode: generated orders on a real-time clock, served on 127.0.0.1.")
ap.add_argument("config", nargs="?", default="configs/sim.yaml")
ap.add_argument("--policy", default="static", choices=["static", *METHODS])
ap.add_argument("--seed", type=int, help="override generator.seed")
ap.add_argument("--speed", type=float, help="initial speed (one of live.speeds); default live.speed")
ap.add_argument("--host", default=None, help=f"must be {L.LOCAL_HOST}")
ap.add_argument("--port", type=int, help="default live.port; 0 = any free port")
ap.add_argument("--hours", type=float, help="sim hours to run (default: the whole window)")
ap.add_argument("--sku-to-face", choices=["synthetic", "real"], default="real",
                help="SKU -> face layout: real = last item picked at each face before the run start (default)")
args = ap.parse_args()

cfg = load_config(args.config)
lc, gcfg = cfg["live"], dict(cfg["generator"])
if args.seed is not None:
    gcfg["seed"] = args.seed
gcfg["sku_to_face"] = args.sku_to_face
host = args.host or lc["host"]
if host != L.LOCAL_HOST:
    raise SystemExit(f"not running: live mode binds to {L.LOCAL_HOST} only, not {host!r}")
try:
    leg_model = E.make_leg_model(cfg)
except ValueError as exc:
    raise SystemExit(f"not running: {exc}")

net = load_travel(cfg["travel"]["dir"])
net.require_stops(cfg["trip_ends"]["start_stop"])
start, end = E.window_bounds(cfg)
days = (end - start).days
profile_path = Path(gcfg["profile_path"])
if not profile_path.exists():
    raise SystemExit(f"{profile_path} not found: run scripts/fit_generator_profile.py first")
profile = json.loads(profile_path.read_text(encoding="utf-8"))
try:
    source = G.make_generator_source(gcfg, cfg["scope"], net, profile, start, days,
                                     dispatch_stop=cfg["trip_ends"]["start_stop"])
except ValueError as exc:
    raise SystemExit(f"not running: {exc}")

rs_info, rs_files, move_cost = None, [], None
if args.policy == "static":
    policy = make_policy("static")
else:
    from reslot.sim import racksmart as RS
    policy, rs_info, rs_files = RS.load_racksmart(cfg, net, method=args.policy)
    move_cost = policy.move_cost
try:                                              # the no-lookahead guard, as replay applies it
    planning = E.check_planning_inputs(policy, start, cfg.get("lookahead_exceptions"))
except ValueError as exc:
    raise SystemExit(f"not running: {exc}")

speed = float(args.speed if args.speed is not None else lc["speed"])
until_s = args.hours * 3600 if args.hours else days * 86400.0
try:
    run = L.LiveRun(source, policy, net, leg_model, cfg["scope"], cfg["trip_ends"], cfg["operators"], speed,
                    lc["speeds"], until_s=until_s, move_cost=move_cost,
                    max_events_per_second=lc["max_events_per_second"])
except ValueError as exc:
    raise SystemExit(f"not running: {exc}")
server = L.serve(run, host, args.port if args.port is not None else lc["port"], REPO / "ui")
port = server.server_address[1]
print(f"live: {source.label}, layout {gcfg['sku_to_face']}, routing {source.routing_rule}, "
      f"policy {policy.name}, seed {gcfg['seed']}, speed {speed:g}x, "
      f"from {start} for {until_s / 3600:g} sim hours")
print(f"serving http://{L.LOCAL_HOST}:{port}/  (events /events, state /state, control POST /control); Ctrl+C to stop",
      flush=True)

run.start()
try:
    while not run.wait(0.5):
        pass
except KeyboardInterrupt:
    print("\nstopping ...", flush=True)
    run.stop()
finally:
    server.shutdown()
    server.server_close()

stamp = run.wall_start.astimezone().strftime("%Y%m%dT%H%M%S")
out = Path(cfg["paths"]["out"]) / f"live_{stamp}"
k = 1
while out.exists():
    k += 1
    out = Path(cfg["paths"]["out"]) / f"live_{stamp}_{k}"
extra = {**run.manifest_extra(), "seed": int(gcfg["seed"]), "planning_inputs": planning,
         "profile": str(profile_path), "server": {"host": L.LOCAL_HOST, "port": port}}
if rs_info is not None:
    extra["notebook_plan"] = rs_info
inputs = [profile_path, net.sources["distance_table"], net.sources["crosswalk"], *rs_files,
          *getattr(leg_model, "files", [])]
manifest = write_manifest(out, {**cfg, "generator": gcfg}, inputs=inputs, extra=extra)

t = extra["totals"]
print(f"sim time covered {extra['sim_time_covered']['from']} .. {extra['sim_time_covered']['to']}; "
      f"{t['orders_arrived']:,} orders arrived, {t['orders_done']:,} done, {t['picks_done']:,} picks, "
      f"{t['pick_feet']:,.0f} ft, {t['operator_hours']:,.1f} operator-hours")
print(f"controls logged: {len(extra['source']['controls'])}; live mode is never citable")
print(f"wrote {out}/manifest.json")
