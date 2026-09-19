"""Manually zip current results for one or more benchmarks into
experiment_1/checkpoints/, regardless of whether a run has finished. Safe to
run at any time, e.g. right before stopping a run early.

Usage:
    python experiment_1/checkpoint_now.py --model qwen3vl
    python experiment_1/checkpoint_now.py --model qwen3vl --dataset hallusionbench
    python experiment_1/checkpoint_now.py --model internvl3_5 --config experiment_1/config.yaml --dataset mathvista chartqa
"""

import argparse

from lib.checkpoint import zip_all_configured_datasets
from lib.config import DATASET_PRESETS, MODEL_PRESETS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--model", required=True, choices=sorted(MODEL_PRESETS),
        help="Which model's results to checkpoint.",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=sorted(DATASET_PRESETS),
        choices=sorted(DATASET_PRESETS),
        help="Benchmarks to checkpoint now (default: all three).",
    )
    args = parser.parse_args()

    zipped = zip_all_configured_datasets(args.config, args.dataset, args.model)
    if not zipped:
        print("Nothing to checkpoint yet (no results found for the requested dataset(s)).")
    for path in zipped:
        print(f"Checkpointed: {path}")


if __name__ == "__main__":
    main()
