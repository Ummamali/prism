"""KAGGLE MULTI-GPU HELPER (one of three - see below). Run several benchmark
shards sequentially in ONE process on ONE GPU, loading the model once and
reusing it across benchmarks. Use this instead of launching inference.py
separately per benchmark whenever a single GPU is assigned more than one
shard - it avoids paying the model-load cost once per benchmark, and each
shard is checkpointed the same way inference.py's run_shard() checkpoints a
single-dataset run.

Where this fits among the multi-GPU scripts:
  - inference.py        : run ONE benchmark shard directly (single GPU, single call)
  - run_multi_shard.py  : (this file) run several shards, ANY mix of benchmarks,
                           in ONE process per GPU - you control exactly how work
                           is split; best when you want max GPU utilization and
                           don't mind assigning shards by hand.
  - run_all_benchmarks.py : a turnkey orchestrator that runs all three benchmarks
                           one at a time, each split 50/50 across both GPUs
                           automatically - simpler to launch, but reloads the
                           model at each benchmark boundary. See that file's
                           docstring for the trade-off.

Usage (example: this GPU handles all of mathvista plus half of hallusionbench):
    python experiment_1/run_multi_shard.py --device cuda:0 \
        --shard mathvista:0:100 --shard hallusionbench:0:50
"""

import argparse

from lib.config import DATASET_PRESETS, load_config
from lib.model import load_model
from inference import run_shard


def parse_shard(spec: str) -> tuple[str, int, int]:
    try:
        dataset, start, end = spec.split(":")
        return dataset, int(start), int(end)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"--shard must look like dataset:start:end (got {spec!r})"
        ) from e


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", required=True, help="e.g. cuda:0 or cuda:1")
    parser.add_argument(
        "--shard",
        action="append",
        required=True,
        dest="shards",
        type=parse_shard,
        help="dataset:start:end, repeatable, e.g. --shard mathvista:0:100",
    )
    args = parser.parse_args()

    for dataset_key, _, _ in args.shards:
        if dataset_key not in DATASET_PRESETS:
            raise ValueError(
                f"Unknown dataset {dataset_key!r} in --shard; expected one of {sorted(DATASET_PRESETS)}"
            )

    # Model config (name/dtype/device/max_new_tokens/...) is shared across all
    # presets, so any shard's config works to load it - only load_model's
    # output is reused across shards, never the dataset-specific cfg.
    bootstrap_cfg = load_config(args.config, dataset=args.shards[0][0])
    print(f"Loading model {bootstrap_cfg['model']['name']} on {args.device} ...")
    loaded = load_model(bootstrap_cfg, device_override=args.device)
    print(f"Model loaded on {loaded.device}. Processing {len(args.shards)} shard(s).")

    for dataset_key, start, end in args.shards:
        cfg = load_config(args.config, dataset=dataset_key)
        print(f"\n=== {dataset_key} [{start}:{end}] on {args.device} ===")
        run_shard(loaded, cfg, dataset_key, start=start, end=end)

    print("\nAll assigned shards complete.")


if __name__ == "__main__":
    main()
