"""One command: (refit + review) -> sim for every policy -> UI bundle.

Run from the repo folder:

  python scripts/run_all.py [--refit DIR [--refit-config FILE]] [--config configs/sim.yaml]
                            [--policies static racksmart ...] [--out FILE] [--sensitivity]

Steps, stopping at the first problem:
  1. --refit DIR   (optional) run DIR/scripts/refit_travel_feet.py with
                   DIR/configs/travel.yaml -- DIR is a checkout of branch
                   travel/refit-feet. It writes the feet fit and its 2026-11-02
                   review outcome into this repo's data/ and runs/ folders.
                   Leave it out once the refit has been run.
  2. check that the leg-time model in the config loads. network_feet loads only
     if the refit's review outcome is "adopted"; if it doesn't, nothing runs.
  3. scripts/run_sim.py once per policy (default: static, abc_cost_test, family_grouping,
     racksmart -- the three UI tabs plus the static reference), same config.
  4. scripts/export_ui.py over those runs -> the bundle ui/index.html opens.
  5. --sensitivity (optional): each notebook method again with
     --move-cases pallet_fill (snapshot-free move cost; folders end _pallet_fill,
     not bundled), then scripts/compare_runs.py: did the ranking change?

Every step is the normal script, run as its own process, so each run writes
its usual outputs and manifest.
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from reslot.config import load_config
from reslot.sim import engine as E
from reslot.sim.racksmart import METHODS

POLICIES = ["static", *METHODS]
# Shown in the UI: ABC (with cost test), family grouping, RackSmart. Static runs as the
# hidden reference every comparison is made against.
DEFAULT_POLICIES = ["static", "abc_cost_test", "family_grouping", "racksmart"]

ap = argparse.ArgumentParser(description="Refit (optional), simulate every policy, and bundle for the UI.")
ap.add_argument("--refit", metavar="DIR", help="checkout of travel/refit-feet to run the refit + review from")
ap.add_argument("--refit-config", metavar="FILE", help="travel config for the refit (default: DIR/configs/travel.yaml)")
ap.add_argument("--config", default="configs/sim.yaml")
ap.add_argument("--policies", nargs="+", default=DEFAULT_POLICIES, choices=POLICIES)
ap.add_argument("--out", help="bundle path (default: <paths.out>/bundle_<start>_<end>.json)")
ap.add_argument("--sensitivity", action="store_true",
                help="also run each notebook method with move_cases pallet_fill and compare the rankings")
a = ap.parse_args()


def step(title, cmd):
    print(f"\n=== {title}\n$ {' '.join(map(str, cmd))}", flush=True)
    r = subprocess.run([str(c) for c in cmd])
    if r.returncode != 0:
        raise SystemExit(f"\nstopped: {title} failed (exit {r.returncode})")


# 1. refit + review
if a.refit:
    d = Path(a.refit)
    script = d / "scripts" / "refit_travel_feet.py"
    tcfg = Path(a.refit_config) if a.refit_config else d / "configs" / "travel.yaml"
    if not (script.exists() and tcfg.exists()):
        raise SystemExit(f"{d} is not a checkout of travel/refit-feet (no {script.name} / {tcfg.name})")
    step("refit on network feet + 2026-11-02 review", [sys.executable, script, tcfg])

# 2. the leg-time model must load before anything is simulated
cfg = load_config(a.config)
print("\n=== checking the leg-time model", flush=True)
try:
    model = E.make_leg_model(cfg)
except ValueError as exc:
    raise SystemExit(f"stopped: {exc}")
review = getattr(model, "provenance", {}).get("review", {})
print(f"leg-time model {model.name} loads (review outcome: {review.get('outcome', 'n/a')})")

# 3. one sim run per policy
w = cfg["window"]
tag = f"{w['start'].replace('-', '')}_{w['end'].replace('-', '')}"
runs = []
for i, pol in enumerate(a.policies, start=1):
    step(f"sim {i}/{len(a.policies)}: {pol}", [sys.executable, REPO / "scripts" / "run_sim.py", a.config, "--policy", pol])
    run_dir = Path(cfg["paths"]["out"]) / E.run_dir_name(cfg, pol, model.name)
    if not (run_dir / "manifest.json").exists():
        raise SystemExit(f"stopped: expected {run_dir} after the {pol} run")
    runs.append(run_dir)

# 4. bundle
cmd = [sys.executable, REPO / "scripts" / "export_ui.py", *runs, "--travel", cfg["travel"]["dir"],
       "--out", a.out or str(Path(cfg["paths"]["out"]) / f"bundle_{tag}.json")]
step("bundle for the UI", cmd)

# 5. move-cost sensitivity
if a.sensitivity:
    methods = [p for p in a.policies if p != "static"]
    for i, pol in enumerate(methods, start=1):
        step(f"sensitivity {i}/{len(methods)}: {pol}, move_cases pallet_fill",
             [sys.executable, REPO / "scripts" / "run_sim.py", a.config, "--policy", pol, "--move-cases", "pallet_fill"])
    out = cfg["paths"]["out"]
    step("compare: inventory snapshot vs pallet fill",
         [sys.executable, REPO / "scripts" / "compare_runs.py", "--a", out, "--b", out, "--b-move-cases", "pallet_fill",
          "--labels", "snapshot", "pallet_fill", "--config", a.config])
print("\ndone: open ui/index.html and choose the bundle above.")
