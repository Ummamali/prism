"""Run exactly ONE benchmark, split across both GPUs via two inference.py
subprocesses. Each GPU half runs a separate inference.py process; once both
finish (i.e. every requested example has been evaluated), automatically runs
decomposition.py (STAGE 3) for that benchmark, then checkpoints (zips) the
combined results.

Keeps experiment_1/checkpoints/run_status.json updated with which benchmark
is currently running, and prints one status line every few seconds with the
last log line of each GPU, so it can be followed from a Kaggle cell's output.

Meant to be run in a Kaggle notebook cell (works with Save & Run):

    !python experiment_1/run_benchmark.py --dataset mathvista --n-samples 50

Usage:
    python experiment_1/run_benchmark.py --dataset chartqa [--n-samples 100] [--max-new-tokens 256] [--config experiment_1/config.yaml]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from lib.checkpoint import checkpoint_dir, zip_dataset_results
from lib.config import DATASET_PRESETS, load_config

LOG_DIR = Path("/kaggle/working")
STATUS_PATH = checkpoint_dir() / "run_status.json"
STATUS_INTERVAL_S = 3
MAX_LINE_CHARS = 160


def write_status(**fields) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": datetime.now().isoformat(timespec="seconds"), **fields}
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def last_log_line(path: Path) -> str:
    """Last non-empty line of a log file. Splits on CR as well as LF so a
    tqdm bar (which redraws in place with CR) yields its latest state."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 4096))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return "(no log yet)"
    lines = [ln.strip() for ln in re.split(r"[\r\n]+", tail) if ln.strip()]
    if not lines:
        return "(no output yet)"
    line = lines[-1]
    return line if len(line) <= MAX_LINE_CHARS else line[: MAX_LINE_CHARS - 3] + "..."


def run_benchmark(
    benchmark: str,
    config_path: Optional[str],
    n_samples: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
) -> None:
    cfg = load_config(config_path, dataset=benchmark)
    if n_samples is not None:
        cfg["data"]["n_samples"] = n_samples
    n = cfg["data"]["n_samples"]
    mid = n // 2  # always an even 50/50 split of however many examples this run uses

    write_status(benchmark=benchmark, status="running", gpu0_range=[0, mid], gpu1_range=[mid, n])
    print(f"\n=== {benchmark}: cuda:0[0:{mid}] + cuda:1[{mid}:{n}] ===", flush=True)

    log0_path = LOG_DIR / f"gpu0_{benchmark}.log"
    log1_path = LOG_DIR / f"gpu1_{benchmark}.log"
    log0 = open(log0_path, "w")
    log1 = open(log1_path, "w")
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    token_args = ["--max-new-tokens", str(max_new_tokens)] if max_new_tokens is not None else []

    p0 = subprocess.Popen(
        [sys.executable, "experiment_1/inference.py", "--dataset", benchmark,
         "--device", "cuda:0", "--start", "0", "--end", str(mid), *token_args],
        stdout=log0, stderr=subprocess.STDOUT, start_new_session=True, env=env,
    )
    p1 = subprocess.Popen(
        [sys.executable, "experiment_1/inference.py", "--dataset", benchmark,
         "--device", "cuda:1", "--start", str(mid), "--end", str(n), *token_args],
        stdout=log1, stderr=subprocess.STDOUT, start_new_session=True, env=env,
    )

    while p0.poll() is None or p1.poll() is None:
        print(
            f"{benchmark}: cuda0: {last_log_line(log0_path)} | cuda1: {last_log_line(log1_path)}",
            flush=True,
        )
        time.sleep(STATUS_INTERVAL_S)

    log0.close()
    log1.close()

    ok = p0.returncode == 0 and p1.returncode == 0

    decomp_ok = None
    if ok:
        decomp_log = open(LOG_DIR / f"decomposition_{benchmark}.log", "w")
        decomp_args = [sys.executable, "experiment_1/decomposition.py", "--dataset", benchmark]
        if config_path is not None:
            decomp_args += ["--config", config_path]
        decomp = subprocess.run(decomp_args, stdout=decomp_log, stderr=subprocess.STDOUT)
        decomp_log.close()
        decomp_ok = decomp.returncode == 0
        if not decomp_ok:
            print(
                f"[WARN] decomposition.py failed for {benchmark} (exit code "
                f"{decomp.returncode}); check decomposition_{benchmark}.log.",
                flush=True,
            )

    zip_path = zip_dataset_results(cfg, benchmark, label="complete")
    write_status(
        benchmark=benchmark,
        status="benchmark_done" if ok else "benchmark_failed",
        gpu0_exit=p0.returncode,
        gpu1_exit=p1.returncode,
        decomposition_ok=decomp_ok,
        checkpoint=str(zip_path) if zip_path else None,
    )
    print(
        f"=== {benchmark} done (exit codes {p0.returncode}, {p1.returncode}) "
        f"-> checkpoint: {zip_path} ===",
        flush=True,
    )
    if not ok:
        print(
            f"[WARN] {benchmark} had a non-zero exit code; check gpu0_{benchmark}.log / "
            f"gpu1_{benchmark}.log before trusting its results.",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", required=True, choices=sorted(DATASET_PRESETS),
        help="Which single benchmark to run.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=None,
        help="How many examples to run for this benchmark. Overrides config's data.n_samples "
             "(note: data_loading.py must have cached at least this many examples already).",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=None,
        help="Cap on generated tokens per trajectory. Overrides config's model.max_new_tokens "
             "(e.g. a small value for a quick smoke test).",
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    run_benchmark(args.dataset, args.config, n_samples=args.n_samples, max_new_tokens=args.max_new_tokens)
    write_status(benchmark=None, status="all_done")
    print(f"\n{args.dataset} complete.", flush=True)


if __name__ == "__main__":
    main()
