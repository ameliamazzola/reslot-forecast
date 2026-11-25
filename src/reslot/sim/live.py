"""Live mode: the engine on a real-time clock, fed by GeneratorSource, served
over HTTP on 127.0.0.1 only. Never citable (manifest mode: live).

  LiveEnvironment  simpy.rt.RealtimeEnvironment(strict=False) that waits for the
                   next event in short slices, so the speed can change and a
                   stop can land while it waits; the clock is re-anchored at a
                   speed change so sim time never jumps
  LiveRun          one policy, one source: runs engine.simulate in a thread,
                   keeps the running counters, publishes snapshots at most
                   max_events_per_second (wall clock, whatever the speed)
  serve            standard-library HTTP server: ui/ as static files,
                   GET /events (Server-Sent Events), GET /state, POST /control

Every speed and affinity change goes through the source's control log
(GeneratorSource.set_speed / set_affinity) with its sim time, so it reaches
the live manifest.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd
import simpy.rt
from simpy.core import EmptySchedule, Infinity

LOCAL_HOST = "127.0.0.1"
SLICE_S = 0.05                     # longest uninterrupted wall-clock wait


class Stopped(Exception):
    pass


class LiveEnvironment(simpy.rt.RealtimeEnvironment):
    """RealtimeEnvironment (strict=False) with a changeable speed and a stop.
    speed = sim seconds per wall second; factor = 1 / speed."""

    def __init__(self, speed: float, until_s: float | None = None):
        super().__init__(initial_time=0, factor=1.0 / float(speed), strict=False)
        self.until_s = until_s
        self.ended_at: float | None = None       # sim time the run ended at, fixed once it has
        self.stop_flag = threading.Event()
        self._lock = threading.Lock()

    @property
    def speed(self) -> float:
        return 1.0 / self.factor

    def clock(self) -> float:
        """Sim time as the wall clock sees it now: between env.now and the next event.
        Once the run has ended, the sim time it ended at."""
        if self.ended_at is not None:
            return self.ended_at
        with self._lock:
            t = self.env_start + (time.monotonic() - self.real_start) / self.factor
        nxt = self.peek()
        cap = nxt if nxt != Infinity else t
        if self.until_s is not None:
            cap = min(cap, self.until_s)
        return max(self.now, min(t, cap))

    def set_speed(self, speed: float) -> None:
        """Re-anchor the clock at the current sim time, then change the rate."""
        if not speed > 0:
            raise ValueError("speed must be > 0")
        now_sim = self.clock()
        with self._lock:
            self.env_start, self.real_start, self._factor = now_sim, time.monotonic(), 1.0 / float(speed)

    def step(self) -> None:
        evt = self.peek()
        if evt is Infinity:
            raise EmptySchedule
        while True:
            if self.stop_flag.is_set():
                raise Stopped
            with self._lock:
                delta = self.real_start + (evt - self.env_start) * self.factor - time.monotonic()
            if delta <= 0:
                break
            time.sleep(min(delta, SLICE_S))
        simpy.core.Environment.step(self)

    def run(self, until=None):
        """Until the schedule is empty, until_s is reached, or stop()."""
        end = None
        try:
            while True:
                nxt = self.peek()
                if self.until_s is not None and nxt >= self.until_s:
                    end = self.until_s                # reached the requested sim span
                    break
                if nxt is Infinity:
                    break
                self.step()
        except Stopped:
            end = self.clock()                        # stopped by hand: where the clock was
        except EmptySchedule:
            pass
        self.ended_at = max(self.now, end if end is not None else self.now)


class LiveRun:
    """One live run: source + policy on a LiveEnvironment, with running counters."""

    def __init__(self, source, policy, net, leg_model, scope: dict, trip_ends: dict, operators: int,
                 speed: float, speeds: list, until_s: float | None = None, move_cost=None,
                 max_events_per_second: float = 4.0, *, arrival_cutoff_s: float | None = None):
        if float(speed) not in [float(s) for s in speeds]:
            raise ValueError(f"speed {speed} is not one of {speeds}")
        self.source, self.policy, self.net, self.leg_model = source, policy, net, leg_model
        self.scope, self.trip_ends, self.operators = scope, trip_ends, int(operators)
        self.speeds, self.move_cost = [float(s) for s in speeds], move_cost
        self.min_interval = 1.0 / float(max_events_per_second)
        self.env = LiveEnvironment(speed, until_s)
        self.arrival_cutoff_s = arrival_cutoff_s
        self.start_ts = pd.Timestamp(source.start)
        self.speeds_used = [{"sim_time_s": 0.0, "speed": float(speed)}]
        self._lock = threading.Lock()
        self._c = {"arrived": 0, "started": 0, "done": 0, "picks": 0, "pick_feet": 0.0,
                   "busy_done_s": 0.0, "wait_s": 0.0}
        self._in_progress: dict = {}           # tour_id -> start time
        self._stop_picks: dict = {}            # stop -> picks since the last published snapshot
        self._subs: list = []
        self._seq = 0
        self.latest: dict = {}
        self.error: BaseException | None = None
        self.result = None
        self.wall_start = self.wall_end = None
        self.finished = threading.Event()

    # -- engine observer ----------------------------------------------------------
    def _observe(self, event: str, p: dict) -> None:
        with self._lock:
            c = self._c
            if event == "release":
                c["arrived"] += 1
            elif event == "start":
                c["started"] += 1
                self._in_progress[p["tour_id"]] = p["t"]
            elif event == "first":
                c["picks"] += 1
                self._stop_picks[p["stop"]] = self._stop_picks.get(p["stop"], 0) + 1
            elif event == "leg":
                c["picks"] += 1
                c["pick_feet"] += p["feet"]
                self._stop_picks[p["to_stop"]] = self._stop_picks.get(p["to_stop"], 0) + 1
            elif event == "done":
                c["done"] += 1
                c["busy_done_s"] += p["t"] - self._in_progress.pop(p["tour_id"])

    # -- snapshots ------------------------------------------------------------------
    def snapshot(self, reset_picks: bool = False) -> dict:
        t = self.env.clock()
        with self._lock:
            c = dict(self._c)
            busy_now = sum(max(0.0, t - s) for s in self._in_progress.values())
            picks = dict(self._stop_picks)
            if reset_picks:
                self._stop_picks = {}
            self._seq += 1
            seq = self._seq
        return {"seq": seq, "policy": self.policy.name, "sim_time_s": t,
                "sim_clock": str(self.start_ts + pd.Timedelta(seconds=t)),
                "speed": self.env.speed, "affinity": float(self.source.gen.affinity),
                "orders_arrived": c["arrived"], "orders_in_progress": c["started"] - c["done"],
                "orders_done": c["done"], "queue_length": c["arrived"] - c["started"],
                "operators_busy": c["started"] - c["done"], "operators": self.operators,
                "picks_by_stop": picks, "picks_done": c["picks"], "pick_feet": c["pick_feet"],
                "operator_hours": (c["busy_done_s"] + busy_now) / 3600,
                "running": not self.finished.is_set()}

    def subscribe(self) -> queue.Queue:
        q = queue.Queue(maxsize=100)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def _publish(self, snap: dict) -> None:
        self.latest = snap
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(snap)
            except queue.Full:                       # a slow client misses snapshots, never blocks the run
                pass

    def _publisher(self) -> None:
        while not self.finished.is_set():
            self._publish(self.snapshot(reset_picks=True))
            self.finished.wait(self.min_interval)
        self._publish(self.snapshot(reset_picks=True))
        for q in list(self._subs):
            try:
                q.put_nowait(None)                   # end of stream
            except queue.Full:
                pass

    # -- controls --------------------------------------------------------------------
    def set_speed(self, speed: float) -> None:
        if float(speed) not in self.speeds:
            raise ValueError(f"speed {speed} is not one of {self.speeds}")
        t = self.env.clock()
        self.env.set_speed(float(speed))
        self.source.set_speed(float(speed), sim_time=t)
        self.speeds_used.append({"sim_time_s": t, "speed": float(speed)})

    def set_affinity(self, value: float) -> None:
        if not 0 <= float(value) <= 1:
            raise ValueError("affinity must be between 0 and 1")
        self.source.set_affinity(float(value), sim_time=self.env.clock())

    # -- run ---------------------------------------------------------------------------
    def _run(self) -> None:
        from .engine import simulate
        try:
            self.result = simulate(self.source, self.operators, self.policy, self.net, self.leg_model,
                                   self.scope, self.trip_ends, move_cost=self.move_cost, env=self.env,
                                   observer=self._observe, arrival_cutoff_s=self.arrival_cutoff_s)
        except BaseException as exc:            # surfaced by wait()/stop()
            self.error = exc
        finally:
            self.wall_end = datetime.now(timezone.utc)
            self.finished.set()

    def start(self) -> None:
        self.wall_start = datetime.now(timezone.utc)
        self.env.sync()
        self._threads = [threading.Thread(target=self._run, daemon=True, name="live-sim"),
                         threading.Thread(target=self._publisher, daemon=True, name="live-publish")]
        for th in self._threads:
            th.start()

    def wait(self, timeout: float | None = None) -> bool:
        done = self.finished.wait(timeout)
        if done and self.error is not None:
            raise self.error
        return done

    def stop(self) -> None:
        self.env.stop_flag.set()
        self.finished.wait()
        for th in getattr(self, "_threads", []):
            th.join(timeout=5)
        if self.error is not None:
            raise self.error

    def totals(self) -> dict:
        s = self.snapshot()
        return {"orders_arrived": s["orders_arrived"], "orders_done": s["orders_done"],
                "orders_in_progress_at_stop": s["orders_in_progress"], "queue_length_at_stop": s["queue_length"],
                "picks_done": s["picks_done"], "pick_feet": s["pick_feet"],
                "operator_hours": s["operator_hours"],
                "sim_time_s": s["sim_time_s"]}

    def manifest_extra(self) -> dict:
        t = self.env.clock()
        return {"mode": "live", "citable": False, "policy": self.policy.name,
                "source": self.source.describe(), "speeds_used": list(self.speeds_used),
                "wall_clock": {"start": str(self.wall_start), "end": str(self.wall_end)},
                "sim_time_covered": {"from": str(self.start_ts), "to": str(self.start_ts + pd.Timedelta(seconds=t)),
                                     "seconds": float(t), "until_s": self.env.until_s,
                                     "stopped_early": bool(self.env.stop_flag.is_set())},
                "operators": self.operators, "leg_time_model": getattr(self.leg_model, "name", None),
                "dispatch_legs": self.trip_ends.get("dispatch_legs"), "drop_off_mode": self.trip_ends.get("drop_off_mode"),
                "totals": self.totals(), "note": "live mode: generated orders on a real-time clock; never citable"}


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

class _Handler(SimpleHTTPRequestHandler):
    run: LiveRun = None

    def log_message(self, *args):                # quiet
        pass

    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/state":
            return self._json(200, self.run.latest or self.run.snapshot())
        if path == "/events":
            return self._events()
        return super().do_GET()

    def _events(self):
        # The stream is the whole connection: once it ends (or the client leaves),
        # the handler closes it instead of waiting for another request on it.
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        q = self.run.subscribe()
        try:
            while True:
                try:
                    snap = q.get(timeout=1.0)
                except queue.Empty:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    continue
                if snap is None:
                    self.wfile.write(b"event: end\ndata: {}\n\n")
                    self.wfile.flush()
                    return
                self.wfile.write(f"data: {json.dumps(snap, default=str)}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        finally:
            self.run.unsubscribe(q)

    def do_POST(self):
        if self.path.split("?")[0] != "/control":
            return self._json(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0) or b"{}")
            if not isinstance(body, dict) or not ({"speed", "affinity"} & set(body)):
                raise ValueError("send {\"speed\": x} and/or {\"affinity\": y}")
            if "speed" in body:
                self.run.set_speed(float(body["speed"]))
            if "affinity" in body:
                self.run.set_affinity(float(body["affinity"]))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, {"ok": True, "controls": self.run.source.controls, "state": self.run.snapshot()})


def serve(run: LiveRun, host: str, port: int, ui_dir) -> ThreadingHTTPServer:
    """Start the server in a background thread. Binds to 127.0.0.1 only."""
    if host != LOCAL_HOST:
        raise ValueError(f"live mode binds to {LOCAL_HOST} only, not {host!r}")
    handler = type("LiveHandler", (_Handler,), {"run": run})
    server = ThreadingHTTPServer((LOCAL_HOST, int(port)), partial(handler, directory=str(Path(ui_dir))))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True, name="live-http").start()
    return server


# --------------------------------------------------------------------------
# live vs replay on one measurement boundary (scripts/live_check.py)
# --------------------------------------------------------------------------
# Boundary rule (decisions 2026-11-20): arrivals stop at the cutoff and both
# sides drain. A tour counts, whole, if and only if it was released in
# [window start, window start + cutoff); every counted tour is run to completion;
# nothing released at or after the cutoff counts. Feet = pick-to-pick leg feet,
# seconds = operator busy time; reported per SKU line (primary), stop visit,
# tour and pick row.

def boundary_rule(start, cutoff_s: float) -> str:
    start = pd.Timestamp(start)
    end = start + pd.Timedelta(seconds=float(cutoff_s))
    same_day = end.normalize() == start.normalize()
    span = (f"{start:%Y-%m-%d} {start:%H:%M}-{end:%H:%M}" if same_day
            else f"{start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M}")
    return (f"tours released {span} (end exclusive), each run to completion; "
            "tours released at or after the cutoff are not counted")


# Units, primary first. Replay counts scan rows -- one SKU on one order can be
# scanned in several rows at one stop (0-ft legs) -- while a generated order has
# one pick per SKU line, so the comparable unit is the SKU line.
UNITS = [("line", "lines", "SKU line: distinct SKU per tour (PRIMARY)"),
         ("visit", "stop_visits", "stop visit: consecutive picks at one stop, in route order"),
         ("tour", "tours", "tour"),
         ("pick", "picks", "pick row: one scan row")]
PRIMARY_UNIT = "line"


def tours_window_row(tours: pd.DataFrame, cutoff_s: float, side: str, routing: str) -> dict:
    """The engine's tours table on the boundary rule: tours released in [0, cutoff_s),
    whole, however late they finished. Feet and operator busy seconds per SKU line
    (primary), per stop visit, per tour and per pick row."""
    missing = [c for c in ("n_lines", "n_stop_visits") if c not in tours.columns]
    if missing:
        raise ValueError(f"tours table has no {missing} (written before SKU lines were recorded): "
                         "rerun scripts/run_sim.py --policy static")
    d = tours[(tours["arrival_s"] >= 0) & (tours["arrival_s"] < float(cutoff_s))]
    feet, secs = float(d["pick_feet"].sum()), float(d["busy_s"].sum())
    row = {"side": side, "routing": routing, "tours": int(len(d)), "lines": int(d["n_lines"].sum()),
           "stop_visits": int(d["n_stop_visits"].sum()), "picks": int(d["n_picks"].sum()),
           "feet": feet, "operator_hours": secs / 3600,
           "drain_end_s": float(d["end_s"].max()) if len(d) else None}
    for unit, count, _ in UNITS:
        n = row[count]
        row[f"ft_per_{unit}"] = feet / n if n else None
        row[f"s_per_{unit}"] = secs / n if n else None
    return row


def live_window_row(run: LiveRun, side: str, routing: str) -> dict:
    """A finished LiveRun with arrival_cutoff_s, measured on the boundary rule from
    the engine's own tours table. Refuses a run that didn't drain (tours still
    queued or in progress)."""
    if run.arrival_cutoff_s is None or not run.finished.is_set():
        raise ValueError("live_window_row needs a finished run with an arrival cutoff")
    t = run.totals()
    if t["orders_in_progress_at_stop"] or t["queue_length_at_stop"] or t["orders_arrived"] != t["orders_done"]:
        raise ValueError(f"live run did not drain: {t['orders_arrived']} released, {t['orders_done']} done")
    return tours_window_row(run.result.tours, run.arrival_cutoff_s, side, routing)


def replay_window_row(tours: pd.DataFrame, cutoff_s: float, side: str = "replay static",
                      routing: str = "actual scan order") -> dict:
    """Replay's tours.csv (every tour already run to completion) on the boundary rule."""
    return tours_window_row(tours, cutoff_s, side, routing)
