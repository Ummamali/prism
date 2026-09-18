"""Run exactly ONE benchmark, split across both GPUs via two inference.py
subprocesses - the manual, one-at-a-time counterpart to
run_all_benchmarks.py's "walk through every configured benchmark
unattended" loop. Use this when you want to launch benchmarks yourself, one
by one (e.g. picking a different --n-samples per benchmark, or only running
whichever one you haven't done yet), instead of one long automated run.

Reuses run_all_benchmarks.py's run_benchmark()/write_status() so the two
scripts can never drift apart: same checkpoints/run_status.json shape, same
per-GPU log file naming, same automatic decomposition.py run once all
requested examples are evaluated, same "complete" checkpoint zip at the
end - so monitor.py, plot_results.py, checkpoint_now.py etc. all work
unchanged regardless of which of the two scripts actually produced the
results.

Meant to be launched as a background subprocess from a Kaggle notebook
cell, same trick as run_all_benchmarks.py:

    import subprocess, sys
    subprocess.Popen(
        [sys.executable, "experiment_1/run_benchmark.py",
         "--dataset", "chartqa", "--n-samples", "150"],
        stdout=open("/kaggle/working/orchestrator.log", "w"),
        stderr=subprocess.STDOUT, start_new_session=True,
    )

Then watch it the same way: `!python experiment_1/monitor.py`.

Usage:
    python experiment_1/run_benchmark.py --dataset chartqa [--n-samples 100] [--config experiment_1/config.yaml]
"""

import argparse

from lib.config import DATASET_PRESETS
from run_all_benchmarks import run_benchmark, write_status


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
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    run_benchmark(args.dataset, args.config, n_samples=args.n_samples)
    write_status(benchmark=None, status="all_done")
    print(f"\n{args.dataset} complete.", flush=True)


if __name__ == "__main__":
    main()
