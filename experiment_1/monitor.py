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

Rendering mode is auto-detected from sys.stdout.isatty():
  - real terminal (ssh, Kaggle's console tab): full multi-line view,
    redrawn in place with an ANSI clear each tick.
  - piped/captured stdout (a Jupyter/Kaggle notebook cell run as
    `!python experiment_1/monitor.py`): ANSI clear-screen codes sent from a
    detached subprocess don't reach the notebook frontend, so instead this
    renders one single compact line and rewrites it in place with a bare
    "\r" (carriage return) - the same trick tqdm uses, which notebook output
    panels do honor even for piped/subprocess output.

Usage:
    python experiment_1/monitor.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from lib.checkpoint import LOG_DIR, checkpoint_dir
from lib.config import load_config, resolve_path
from lib.io_utils import load_jsonl

REFRESH_SECONDS = 1
STALL_SECONDS = 30  # no log output in this long -> flag as possibly stuck
IS_TTY = sys.stdout.isatty()


def gpu_status(log_path: Path) -> tuple[str, str]:
    if not log_path.exists():
        return "not started", ""
    try:
        raw = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        raw = ""
    # tqdm progress lines are "\r"-separated, not "\n"-separated; normalize
    # both before splitting so the "last line" is a real last line.
    lines = [ln.strip() for ln in raw.replace("\r", "\n").splitlines() if ln.strip()]
    last_line = lines[-1] if lines else ""

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


def gather() -> dict:
    """Collect the current state as a plain dict; both render modes read
    from this so they can never disagree about the underlying numbers.
    """
    status_path = checkpoint_dir() / "run_status.json"
    if not status_path.exists():
        return {"have_status": False}

    status = json.loads(status_path.read_text(encoding="utf-8"))
    benchmark = status.get("benchmark")
    result = {
        "have_status": True,
        "benchmark": benchmark,
        "overall_status": status.get("status", "unknown"),
    }
    if not benchmark:
        return result

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

    result.update(
        gpu0_stat=gpu0_stat, gpu0_tail=gpu0_tail, gpu0_done=gpu0_done, gpu0_total=gpu0_total,
        gpu1_stat=gpu1_stat, gpu1_tail=gpu1_tail, gpu1_done=gpu1_done, gpu1_total=gpu1_total,
    )
    return result


def render_multiline(state: dict) -> None:
    print("\033[2J\033[H", end="")  # ANSI clear + cursor home
    print(f"PRISM experiment monitor - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if not state["have_status"]:
        print("\nWaiting for run_all_benchmarks.py to start "
              "(no run_status.json yet)...")
        return

    benchmark = state["benchmark"]
    print(f"\n1) Current benchmark: {benchmark or '(none)'}  [{state['overall_status']}]")

    if not benchmark:
        print("\n2) GPU 0: idle")
        print("3) GPU 1: idle")
        print("\n4) Overall progress: n/a")
        return

    print(f"\n2) GPU 0: {state['gpu0_stat']}")
    print(f"   examples evaluated: {state['gpu0_done']}/{state['gpu0_total']}")
    if state["gpu0_tail"]:
        print(f"   last log line: {state['gpu0_tail']}")

    print(f"\n3) GPU 1: {state['gpu1_stat']}")
    print(f"   examples evaluated: {state['gpu1_done']}/{state['gpu1_total']}")
    if state["gpu1_tail"]:
        print(f"   last log line: {state['gpu1_tail']}")

    total_done = state["gpu0_done"] + state["gpu1_done"]
    total = state["gpu0_total"] + state["gpu1_total"]
    pct = (100 * total_done / total) if total else 0.0
    print(f"\n4) Overall progress ({benchmark}): {total_done}/{total} ({pct:.1f}%)")


_last_line_len = 0


def render_single_line(state: dict) -> None:
    global _last_line_len

    if not state["have_status"]:
        line = f"[{time.strftime('%H:%M:%S')}] waiting for run_all_benchmarks.py to start..."
    elif not state["benchmark"]:
        line = f"[{time.strftime('%H:%M:%S')}] benchmark=(none) [{state['overall_status']}] - both GPUs idle"
    else:
        total_done = state["gpu0_done"] + state["gpu1_done"]
        total = state["gpu0_total"] + state["gpu1_total"]
        pct = (100 * total_done / total) if total else 0.0
        line = (
            f"[{time.strftime('%H:%M:%S')}] benchmark={state['benchmark']} "
            f"| GPU0 {state['gpu0_stat']} {state['gpu0_done']}/{state['gpu0_total']} "
            f"| GPU1 {state['gpu1_stat']} {state['gpu1_done']}/{state['gpu1_total']} "
            f"| overall {total_done}/{total} ({pct:.1f}%)"
        )

    # Pad with spaces so a shorter line fully overwrites a longer previous one.
    padded = line.ljust(_last_line_len)
    _last_line_len = len(line)
    print("\r" + padded, end="", flush=True)


def render() -> None:
    state = gather()
    if IS_TTY:
        render_multiline(state)
    else:
        render_single_line(state)


def main() -> None:
    while True:
        render()
        time.sleep(REFRESH_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
