"""Run mathvista, hallusionbench, and chartqa one benchmark at a time, each
split 50/50 across both GPUs in parallel via inference.py subprocesses.
Once both GPU halves of a benchmark finish, checkpoints (zips) that
benchmark's combined results before moving to the next benchmark.

Meant to be launched once as a single background process (see notebook
cell) and left to run unattended. Keeps experiment_1/checkpoints/run_status.json
updated with which benchmark is currently running, for a separate monitor
cell to poll.

Trade-off vs. run_multi_shard.py: this launches a fresh inference.py process
per GPU per benchmark, so the model is reloaded at each benchmark boundary
(~5-10s x 6 loads here) rather than staying resident across benchmarks. That
cost is negligible next to the ~30min total runtime and buys a simpler,
fully-parallel-per-benchmark schedule with a clean checkpoint between each.

Usage:
    python experiment_1/run_all_benchmarks.py [--config experiment_1/config.yaml]
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from checkpoint import zip_dataset_results
from common import EXPERIMENT_DIR, load_config

BENCHMARKS = ["mathvista", "hallusionbench", "chartqa"]
SPLIT = 50  # each benchmark's n_samples examples split evenly across cuda:0 / cuda:1
LOG_DIR = Path("/kaggle/working")
STATUS_PATH = EXPERIMENT_DIR / "checkpoints" / "run_status.json"


def write_status(**fields) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": datetime.now().isoformat(timespec="seconds"), **fields}
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_benchmark(benchmark: str, config_path: Optional[str]) -> None:
    cfg = load_config(config_path, dataset=benchmark)
    n = cfg["data"]["n_samples"]
    mid = min(SPLIT, n)

    write_status(benchmark=benchmark, status="running", gpu0_range=[0, mid], gpu1_range=[mid, n])
    print(f"\n=== {benchmark}: cuda:0[0:{mid}] + cuda:1[{mid}:{n}] ===", flush=True)

    log0 = open(LOG_DIR / f"gpu0_{benchmark}.log", "w")
    log1 = open(LOG_DIR / f"gpu1_{benchmark}.log", "w")

    p0 = subprocess.Popen(
        [sys.executable, "experiment_1/inference.py", "--dataset", benchmark,
         "--device", "cuda:0", "--start", "0", "--end", str(mid)],
        stdout=log0, stderr=subprocess.STDOUT, start_new_session=True,
    )
    p1 = subprocess.Popen(
        [sys.executable, "experiment_1/inference.py", "--dataset", benchmark,
         "--device", "cuda:1", "--start", str(mid), "--end", str(n)],
        stdout=log1, stderr=subprocess.STDOUT, start_new_session=True,
    )

    while p0.poll() is None or p1.poll() is None:
        time.sleep(5)

    log0.close()
    log1.close()

    ok = p0.returncode == 0 and p1.returncode == 0
    zip_path = zip_dataset_results(cfg, benchmark, label="complete")
    write_status(
        benchmark=benchmark,
        status="benchmark_done" if ok else "benchmark_failed",
        gpu0_exit=p0.returncode,
        gpu1_exit=p1.returncode,
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
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    for benchmark in BENCHMARKS:
        run_benchmark(benchmark, args.config)

    write_status(benchmark=None, status="all_done")
    print("\nAll benchmarks complete.", flush=True)


if __name__ == "__main__":
    main()
