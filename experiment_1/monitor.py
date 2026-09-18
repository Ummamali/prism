"""Live terminal/notebook monitor for run_all_benchmarks.py AND plain local
runs of inference.py. Refreshes once a second and reports:
  1) Which benchmark is currently running
  2) GPU 0's status + how many of its assigned examples are done
  3) GPU 1's status + how many of its assigned examples are done
  4) Overall progress across both GPUs for the current benchmark
  5) Per in-flight example: which pid, which STAGE it's at (real/blind
     trajectory generation, or "score step s/N Ck"), and how long it's
     been there - so a single slow/stuck example shows up as "stage=score
     step 7/23 C3, elapsed=340s" instead of the whole run just looking
     frozen with no indication of where.

Reads two kinds of files:
  - Files run_all_benchmarks.py / inference.py already wrote before this:
      checkpoints/run_status.json  -> current benchmark, each GPU's index range
      results_{benchmark}/per_example/{pid}.json  -> one file per finished example
      /kaggle/working/gpu{0,1}_{benchmark}.log     -> per-GPU subprocess output
        (mtime recency + last line -> running / stalled / finished)
  - Section 5's per-example stage data: checkpoints/progress_{device}.json,
    written by lib/progress.py (see that module for the stage names) from
    inside generate_trajectory() and inference.py's scoring loop.

No run_status.json (i.e. no run_all_benchmarks.py orchestrator - just a
plain `python inference.py`)? Falls back to whatever dataset config.yaml
itself points at for sections 1-4, so a local single-process run is still
monitorable. Section 5 needs no benchmark at all - it just shows whatever
progress_*.json files currently exist.

Rendering mode is auto-detected:
  - running inside a Jupyter/IPython kernel (Kaggle notebook cell run as
    `%run experiment_1/monitor.py`, IN-PROCESS - NOT `!python ...`): full
    multi-line view, redrawn in place via IPython.display.clear_output().
    `!python ...` forks a detached subprocess whose stdout the notebook can
    only append as plain text - there is no way for that subprocess to
    erase previously-printed cell output, which is why that invocation
    scrolls forever instead of updating in place. Use `%run` for monitor.py.
  - real terminal (ssh, Kaggle's console tab): full multi-line view,
    redrawn in place with an ANSI clear each tick.
  - anything else (output piped/redirected to a file, no tty, no IPython):
    one single compact line rewritten in place with a bare "\r" (carriage
    return) - the same trick tqdm uses.

Usage (Kaggle/Jupyter notebook cell):
    %run experiment_1/monitor.py

Usage (terminal):
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
from lib.progress import read_all_progress

REFRESH_SECONDS = 1
STALL_SECONDS = 30  # no log output in this long -> flag as possibly stuck
# A single generation stage or scoring forward pass taking longer than this
# gets flagged in section 5 - a rough heuristic, not a hard timeout. CPU
# runs of even a 2B model can genuinely take a few minutes per stage, so
# treat the flag as "worth a look", not "definitely dead".
STAGE_STALL_SECONDS = 120
IS_TTY = sys.stdout.isatty()


def _in_ipython() -> bool:
    """True only when running IN-PROCESS inside a Jupyter/IPython kernel
    (e.g. via %run). A `!python monitor.py` subprocess does not satisfy
    this even though IPython is installed in that environment too - it has
    no kernel of its own, so get_ipython() returns None there.
    """
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


IS_IPYTHON = _in_ipython()


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
    have_status = status_path.exists()
    status = json.loads(status_path.read_text(encoding="utf-8")) if have_status else {}
    benchmark = status.get("benchmark")

    if not benchmark:
        # No orchestrator (run_all_benchmarks.py) status, or it hasn't
        # picked a benchmark yet - fall back to whatever config.yaml's own
        # data.dataset says, so a plain `python inference.py` run (no
        # orchestrator at all) is still monitorable for sections 1-4.
        try:
            benchmark = load_config()["data"]["dataset"]
        except Exception:
            benchmark = None

    result = {
        "have_status": have_status,
        "benchmark": benchmark,
        "overall_status": status.get("status", "local" if benchmark else "unknown"),
        "progress": read_all_progress(),
    }
    if not benchmark:
        return result

    cfg = load_config(dataset=benchmark)
    cache_path = resolve_path(cfg["data"]["cache_path"])
    raw_dir = resolve_path(cfg["output"]["raw_dir"])
    pids = [ex["pid"] for ex in load_jsonl(cache_path)] if cache_path.exists() else []

    # No orchestrator range info (local run) -> treat the whole cached
    # sample as "GPU 0"'s range and leave "GPU 1" empty, rather than
    # showing a stale/misleading [0, 0].
    gpu0_lo, gpu0_hi = status.get("gpu0_range", [0, len(pids)])
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


def build_progress_lines(state: dict) -> list[str]:
    """Section 5: one line per currently in-flight example (one per active
    worker process/device), showing its pid, current stage, and how long
    it's been sitting at that stage. This is what actually answers "is one
    example too slow, and where is it stuck" - independent of whether a
    benchmark/orchestrator status is known at all.
    """
    lines = ["\n5) In-flight examples (per active worker):"]
    entries = state.get("progress") or []
    if not entries:
        lines.append("   (none reporting - either idle between examples, or an older run "
                      "from before stage tracking was added)")
        return lines

    now = time.time()
    for p in sorted(entries, key=lambda e: e.get("since", now)):
        elapsed = now - p.get("since", now)
        flag = "  <-- possibly stuck, check on it" if elapsed > STAGE_STALL_SECONDS else ""
        lines.append(
            f"   [{p.get('device', '?')}] pid={p.get('pid', '?')} "
            f"stage={p.get('stage', '?')} elapsed={elapsed:.0f}s{flag}"
        )
    return lines


def build_lines(state: dict) -> list[str]:
    lines = [
        f"PRISM experiment monitor - {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
    ]

    if not state["have_status"]:
        lines.append("\n(no run_all_benchmarks.py orchestrator detected - "
                      "showing local config.yaml's dataset, if any)")

    benchmark = state["benchmark"]
    lines.append(f"\n1) Current benchmark: {benchmark or '(none)'}  [{state['overall_status']}]")

    if not benchmark:
        lines.append("\n2) GPU 0: idle")
        lines.append("3) GPU 1: idle")
        lines.append("\n4) Overall progress: n/a")
        lines.extend(build_progress_lines(state))
        return lines

    lines.append(f"\n2) GPU 0: {state['gpu0_stat']}")
    lines.append(f"   examples evaluated: {state['gpu0_done']}/{state['gpu0_total']}")
    if state["gpu0_tail"]:
        lines.append(f"   last log line: {state['gpu0_tail']}")

    lines.append(f"\n3) GPU 1: {state['gpu1_stat']}")
    lines.append(f"   examples evaluated: {state['gpu1_done']}/{state['gpu1_total']}")
    if state["gpu1_tail"]:
        lines.append(f"   last log line: {state['gpu1_tail']}")

    total_done = state["gpu0_done"] + state["gpu1_done"]
    total = state["gpu0_total"] + state["gpu1_total"]
    pct = (100 * total_done / total) if total else 0.0
    lines.append(f"\n4) Overall progress ({benchmark}): {total_done}/{total} ({pct:.1f}%)")

    lines.extend(build_progress_lines(state))
    return lines


def render_multiline(state: dict) -> None:
    print("\033[2J\033[H", end="")  # ANSI clear + cursor home
    print("\n".join(build_lines(state)))


def render_notebook(state: dict) -> None:
    from IPython.display import clear_output
    clear_output(wait=True)
    print("\n".join(build_lines(state)))


_last_line_len = 0


def _progress_suffix(state: dict) -> str:
    """Compact "| stage=... elapsed=Ns[!]" suffix for the single-line
    renderer - the single-line mode is a bare "\\r"-rewritten line (see
    module docstring), so this only ever shows the single
    longest-elapsed in-flight worker rather than one line per worker.
    """
    entries = state.get("progress") or []
    if not entries:
        return ""
    now = time.time()
    worst = max(entries, key=lambda e: now - e.get("since", now))
    elapsed = now - worst.get("since", now)
    flag = "!" if elapsed > STAGE_STALL_SECONDS else ""
    return f" | [{worst.get('device', '?')}] pid={worst.get('pid', '?')} stage={worst.get('stage', '?')} {elapsed:.0f}s{flag}"


def render_single_line(state: dict) -> None:
    global _last_line_len

    if not state["benchmark"]:
        status_bit = "waiting for run_all_benchmarks.py to start..." if not state["have_status"] \
            else f"benchmark=(none) [{state['overall_status']}] - both GPUs idle"
        line = f"[{time.strftime('%H:%M:%S')}] {status_bit}"
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

    line += _progress_suffix(state)

    # Pad with spaces so a shorter line fully overwrites a longer previous one.
    padded = line.ljust(_last_line_len)
    _last_line_len = len(line)
    print("\r" + padded, end="", flush=True)


def render() -> None:
    state = gather()
    if IS_IPYTHON:
        render_notebook(state)
    elif IS_TTY:
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
