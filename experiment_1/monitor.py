"""Live terminal/notebook monitor for run_all_benchmarks.py. Refreshes once
a second and reports:
  1) Which benchmark is currently running
  2) GPU 0's status + how many of its assigned examples are done
  3) GPU 1's status + how many of its assigned examples are done
  4) Overall progress across both GPUs for the current benchmark

Reads only files run_all_benchmarks.py / inference.py already write - no
extra instrumentation needed:
  - checkpoints/run_status.json  -> current benchmark, each GPU's index range
  - results_{benchmark}/per_example/{pid}.json  -> one file per finished example
  - /kaggle/working/gpu{0,1}_{benchmark}.log     -> per-GPU subprocess output
    (mtime recency + last line -> running / stalled / finished)

Usage:
    python experiment_1/monitor.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from lib.checkpoint import LOG_DIR, checkpoint_dir
from lib.config import load_config, resolve_path
from lib.io_utils import load_jsonl

REFRESH_SECONDS = 1
STALL_SECONDS = 30  # no log output in this long -> flag as possibly stuck

try:
    from IPython.display import clear_output

    def _clear() -> None:
        clear_output(wait=True)
except ImportError:
    def _clear() -> None:
        print("\033[2J\033[H", end="")  # ANSI clear + cursor home, for plain terminals


def gpu_status(log_path: Path) -> tuple[str, str]:
    if not log_path.exists():
        return "not started", ""
    try:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        tail = []
    last_line = tail[-1].strip() if tail else ""

    if "shard finished" in last_line or "Per-example results written" in last_line:
        return "finished", last_line

    age = time.time() - log_path.stat().st_mtime
    if age > STALL_SECONDS:
        return f"stalled ({int(age)}s since last output)", last_line
    return "running", last_line


def count_done(raw_dir: Path, pids: list[str], lo: int, hi: int) -> tuple[int, int]:
    wanted = pids[lo:hi]
    if not raw_dir.exists():
        return 0, len(wanted)
    done = sum(1 for pid in wanted if (raw_dir / f"{pid}.json").exists())
    return done, len(wanted)


def render() -> None:
    _clear()
    print(f"PRISM experiment monitor - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    status_path = checkpoint_dir() / "run_status.json"
    if not status_path.exists():
        print("\nWaiting for run_all_benchmarks.py to start "
              "(no run_status.json yet)...")
        return

    status = json.loads(status_path.read_text(encoding="utf-8"))
    benchmark = status.get("benchmark")
    overall_status = status.get("status", "unknown")

    print(f"\n1) Current benchmark: {benchmark or '(none)'}  [{overall_status}]")

    if not benchmark:
        print("\n2) GPU 0: idle")
        print("3) GPU 1: idle")
        print("\n4) Overall progress: n/a")
        return

    cfg = load_config(dataset=benchmark)
    cache_path = resolve_path(cfg["data"]["cache_path"])
    raw_dir = resolve_path(cfg["output"]["raw_dir"])
    pids = [ex["pid"] for ex in load_jsonl(cache_path)] if cache_path.exists() else []

    gpu0_lo, gpu0_hi = status.get("gpu0_range", [0, 0])
    gpu1_lo, gpu1_hi = status.get("gpu1_range", [0, 0])

    gpu0_stat, gpu0_tail = gpu_status(LOG_DIR / f"gpu0_{benchmark}.log")
    gpu1_stat, gpu1_tail = gpu_status(LOG_DIR / f"gpu1_{benchmark}.log")

    gpu0_done, gpu0_total = count_done(raw_dir, pids, gpu0_lo, gpu0_hi)
    gpu1_done, gpu1_total = count_done(raw_dir, pids, gpu1_lo, gpu1_hi)

    print(f"\n2) GPU 0: {gpu0_stat}")
    print(f"   examples evaluated: {gpu0_done}/{gpu0_total}")
    if gpu0_tail:
        print(f"   last log line: {gpu0_tail}")

    print(f"\n3) GPU 1: {gpu1_stat}")
    print(f"   examples evaluated: {gpu1_done}/{gpu1_total}")
    if gpu1_tail:
        print(f"   last log line: {gpu1_tail}")

    total_done = gpu0_done + gpu1_done
    total = gpu0_total + gpu1_total
    pct = (100 * total_done / total) if total else 0.0
    print(f"\n4) Overall progress ({benchmark}): {total_done}/{total} ({pct:.1f}%)")


def main() -> None:
    while True:
        render()
        time.sleep(REFRESH_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
