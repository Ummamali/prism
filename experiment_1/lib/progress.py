"""Live per-example stage tracking, so monitor.py can show not just "how
many examples are done" but "what is the in-flight example doing right
now, and how long has it been there" - the thing you actually need to
diagnose one slow/stuck example instead of staring at a frozen tqdm bar.

Each worker process (one per --device: a plain `python inference.py` run,
or one of run_all_benchmarks.py's/run_multi_shard.py's per-GPU processes)
writes its current (pid, stage, timestamp) to its own
checkpoints/progress_{device}.json every time it moves to a new stage.
One file per device means concurrent GPU0/GPU1 workers never clobber each
other's progress file.

Best-effort by design: a failed progress write must never crash a run, so
every write/read here swallows its own errors.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .checkpoint import checkpoint_dir


def _safe_device_name(device: str) -> str:
    return (device or "unknown").replace(":", "_").replace("/", "_")


def _progress_path(device: str) -> Path:
    return checkpoint_dir() / f"progress_{_safe_device_name(device)}.json"


def report_stage(device: str, pid: str, stage: str) -> None:
    """Record that `pid` (on this process's `device`) has just started
    `stage`. Stage names in use (see inference.py / lib/scoring.py):
      "real:stage1" / "real:stage2"   - real (image-present) trajectory generation
      "blind:stage1" / "blind:stage2" - blind (image-ablated) trajectory generation
      "real:generate" / "blind:generate" - single-call generation (budget_forcing disabled)
      "score step s/N C1".."C4"       - teacher-forced scoring of step s of N under condition C1-C4
    """
    payload = {"device": device, "pid": pid, "stage": stage, "since": time.time()}
    try:
        _progress_path(device).write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass


def clear_progress(device: str) -> None:
    """Remove this device's progress file once its shard finishes, so a
    finished worker doesn't linger in the monitor as a fake "in flight"
    example forever.
    """
    try:
        _progress_path(device).unlink(missing_ok=True)
    except OSError:
        pass


def read_all_progress() -> list[dict]:
    """Every currently-known in-flight example, one entry per active
    worker process/device. No particular ordering guarantee - callers sort
    however they want (e.g. by elapsed time).
    """
    out = []
    for path in sorted(checkpoint_dir().glob("progress_*.json")):
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return out
