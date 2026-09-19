"""Run exactly ONE benchmark. The examples are split evenly across however
many GPUs are visible (one inference.py subprocess per GPU; with a single GPU,
or none, one process runs all of them). Once all processes finish (i.e. every
requested example has been evaluated), automatically runs decomposition.py
(STAGE 3) for that benchmark, then checkpoints (zips) the combined results.

The final zip is written even if a GPU process fails or this script is
interrupted (labelled "incomplete" in that case), and is verified before this
script exits.

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
from typing import Optional
from pathlib import Path

from lib.checkpoint import LOG_DIR, checkpoint_dir, zip_dataset_results
from lib.config import DATASET_PRESETS, load_config, resolve_path
from lib.io_utils import load_jsonl

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


def detect_devices() -> list[str]:
    """One device string per visible CUDA GPU; ["auto"] (inference.py picks
    CPU) if there are none."""
    try:
        import torch

        n_gpus = torch.cuda.device_count()
    except Exception:  # noqa: BLE001 - torch missing/broken: fall back to auto
        n_gpus = 0
    return [f"cuda:{i}" for i in range(n_gpus)] or ["auto"]


def run_benchmark(
    benchmark: str,
    config_path: Optional[str],
    n_samples: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
) -> bool:
    """Returns True if every inference process exited cleanly."""
    cfg = load_config(config_path, dataset=benchmark)
    if n_samples is not None:
        cfg["data"]["n_samples"] = n_samples
    n = cfg["data"]["n_samples"]

    cache_path = resolve_path(cfg["data"]["cache_path"])
    if not cache_path.exists():
        raise SystemExit(
            f"No cached data at {cache_path}. Run: python experiment_1/data_loading.py --dataset {benchmark}"
        )
    n_cached = len(load_jsonl(cache_path))
    if n > n_cached:
        print(f"[WARN] requested {n} examples but only {n_cached} are cached; using {n_cached}.", flush=True)
        n = n_cached
        cfg["data"]["n_samples"] = n

    devices = detect_devices()
    k = len(devices)
    # Contiguous, near-even split of [0, n) across the available devices
    # (a single device gets everything).
    ranges = [(d * n // k, (d + 1) * n // k) for d in range(k)]

    write_status(benchmark=benchmark, status="running", devices=devices, ranges=[list(r) for r in ranges])
    plan = " + ".join(f"{dev}[{a}:{b}]" for dev, (a, b) in zip(devices, ranges))
    print(f"\n=== {benchmark}: {plan} ===", flush=True)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    token_args = ["--max-new-tokens", str(max_new_tokens)] if max_new_tokens is not None else []

    log_paths, logs, procs = [], [], []
    exit_codes: list = []
    ok = False
    decomp_ok = None
    interrupted = False
    try:
        for d, (dev, (a, b)) in enumerate(zip(devices, ranges)):
            log_path = LOG_DIR / f"gpu{d}_{benchmark}.log"
            log = open(log_path, "w")
            cmd = [sys.executable, "experiment_1/inference.py", "--dataset", benchmark,
                   "--start", str(a), "--end", str(b), *token_args]
            if dev != "auto":
                cmd += ["--device", dev]
            procs.append(subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env,
            ))
            log_paths.append(log_path)
            logs.append(log)

        while any(p.poll() is None for p in procs):
            status = " | ".join(f"cuda{d}: {last_log_line(lp)}" for d, lp in enumerate(log_paths))
            print(f"{benchmark}: {status}", flush=True)
            time.sleep(STATUS_INTERVAL_S)

        exit_codes = [p.returncode for p in procs]
        ok = all(rc == 0 for rc in exit_codes)

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
    except BaseException:
        interrupted = True
        for p in procs:
            if p.poll() is None:
                p.terminate()
        raise
    finally:
        for log in logs:
            log.close()

        n_done = len(list(resolve_path(cfg["output"]["raw_dir"]).glob("*.json")))
        status = "interrupted" if interrupted else ("benchmark_done" if ok else "benchmark_failed")
        # Status is written before zipping so the zip's copy of run_status.json is final.
        write_status(
            benchmark=benchmark, status=status, exit_codes=exit_codes,
            decomposition_ok=decomp_ok, examples_done=n_done, examples_requested=n,
        )
        zip_path = zip_dataset_results(cfg, benchmark, label="complete" if ok else "incomplete")
        write_status(
            benchmark=benchmark, status=status, exit_codes=exit_codes,
            decomposition_ok=decomp_ok, examples_done=n_done, examples_requested=n,
            checkpoint=str(zip_path) if zip_path else None,
        )
        zip_info = f"{zip_path}, {zip_path.stat().st_size / 1e6:.1f} MB, verified" if zip_path else "none - no results to zip"
        print(
            f"=== {benchmark} {status} (exit codes {exit_codes}); examples {n_done}/{n} "
            f"-> checkpoint: {zip_info} ===",
            flush=True,
        )
        if not ok:
            print(
                f"[WARN] {benchmark} did not finish cleanly; check the gpu*_{benchmark}.log "
                f"files before trusting its results.",
                flush=True,
            )
        elif n_done < n:
            print(
                f"[WARN] only {n_done}/{n} examples produced results; the rest failed "
                f"(look for '[WARN] example ... failed' in the gpu*_{benchmark}.log files).",
                flush=True,
            )
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", required=True, choices=sorted(DATASET_PRESETS),
        help="Which single benchmark to run.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=None,
        help="How many examples to run for this benchmark. Overrides config's data.n_samples "
             "(capped at what data_loading.py has cached).",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=None,
        help="Cap on generated tokens per trajectory. Overrides config's model.max_new_tokens "
             "(e.g. a small value for a quick smoke test).",
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    ok = run_benchmark(args.dataset, args.config, n_samples=args.n_samples, max_new_tokens=args.max_new_tokens)
    write_status(benchmark=None, status="all_done" if ok else "failed")
    print(f"\n{args.dataset} {'complete' if ok else 'finished with errors'}.", flush=True)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
