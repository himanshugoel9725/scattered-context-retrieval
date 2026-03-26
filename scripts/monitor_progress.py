#!/usr/bin/env python3
"""Monitor experiment progress by reading checkpoint files.

Usage:
    python scripts/monitor_progress.py              # Live refresh every 30s
    python scripts/monitor_progress.py --once       # Single snapshot
    python scripts/monitor_progress.py --exp 1.3    # Monitor exp 1.3
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

EXPERIMENT_TARGETS = {
    "1.2": {
        "file": "exp1_2/exp1_2_results.json",
        "target": 500,
        "parse": lambda d: {
            "localized": len(d.get("localized", [])),
            "scattered": len(d.get("scattered", [])),
        },
    },
    "1.3": {
        "file": "exp1_3/exp1_3_results.json",
        "target": 200,
        "parse": lambda d: {"entities": len(d) if isinstance(d, list) else 0},
    },
}

BAR_WIDTH = 40


def _bar(done: int, total: int, width: int = BAR_WIDTH) -> str:
    frac = min(done / max(total, 1), 1.0)
    filled = int(frac * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {done}/{total} ({frac:.1%})"


def _read_checkpoint(exp_id: str) -> dict | None:
    spec = EXPERIMENT_TARGETS.get(exp_id)
    if not spec:
        return None
    path = RESULTS_DIR / spec["file"]
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        counts = spec["parse"](data)
        mtime = os.path.getmtime(path)
        return {"counts": counts, "total": sum(counts.values()),
                "target": spec["target"], "mtime": mtime}
    except (json.JSONDecodeError, KeyError):
        return None


def display(exp_id: str, prev_total: int | None = None, prev_time: float | None = None):
    info = _read_checkpoint(exp_id)
    if info is None:
        print(f"  Exp {exp_id}: No checkpoint file found yet.")
        return 0, time.time()

    total = info["total"]
    target = info["target"]
    age = time.time() - info["mtime"]

    # Header
    print(f"  Exp {exp_id}  {_bar(total, target)}")

    # Breakdown
    parts = "  ".join(f"{k}: {v}" for k, v in info["counts"].items())
    print(f"    {parts}")

    # Rate estimation
    if prev_total is not None and prev_time is not None:
        dt = time.time() - prev_time
        delta = total - prev_total
        if dt > 0 and delta > 0:
            rate = delta / (dt / 60)
            remaining = target - total
            eta_min = remaining / rate if rate > 0 else float("inf")
            print(f"    Rate: ~{rate:.1f} queries/min  |  ETA: ~{eta_min:.0f} min")

    # File freshness
    if age < 60:
        fresh = f"{age:.0f}s ago"
    elif age < 3600:
        fresh = f"{age / 60:.0f}m ago"
    else:
        fresh = f"{age / 3600:.1f}h ago"
    print(f"    Last checkpoint: {fresh}")

    return total, time.time()


def main():
    parser = argparse.ArgumentParser(description="Monitor experiment progress")
    parser.add_argument("--exp", default="all", help="Experiment ID (1.2, 1.3, or 'all')")
    parser.add_argument("--once", action="store_true", help="Single snapshot, no refresh")
    parser.add_argument("--interval", type=int, default=30, help="Refresh interval in seconds")
    args = parser.parse_args()

    exp_ids = list(EXPERIMENT_TARGETS.keys()) if args.exp == "all" else [args.exp]
    prev = {eid: (None, None) for eid in exp_ids}

    try:
        while True:
            os.system("clear" if os.name != "nt" else "cls")
            print("=" * 60)
            print("  EXPERIMENT PROGRESS MONITOR")
            print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)
            print()

            for eid in exp_ids:
                pt, tt = prev[eid]
                new_total, new_time = display(eid, pt, tt)
                prev[eid] = (new_total, new_time)
                print()

            if args.once:
                break

            print(f"  Refreshing in {args.interval}s... (Ctrl+C to stop)")
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n  Monitor stopped.")


if __name__ == "__main__":
    main()
