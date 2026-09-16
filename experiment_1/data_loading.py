"""STAGE 1 of the pipeline: pull a small pilot sample from the configured
benchmark (data.dataset in config.yaml, or --dataset) and cache it locally
as one .jsonl file (one row per example) plus a folder of saved images.

Every benchmark has a different Hugging Face layout (different dataset id,
split name, and column names for the image/question/answer) - see
lib/config.py's DATASET_PRESETS for that mapping. This script's only job is
to normalize whichever benchmark you picked into one common on-disk format,
so inference.py (STAGE 2) never has to know which benchmark it's scoring.

Usage:
    python experiment_1/data_loading.py [--config experiment_1/config.yaml] [--dataset mathvista|hallusionbench|chartqa]

Saves (paths are {dataset}-templated, see config.yaml):
    experiment_1/data/images_{dataset}/{pid}.png   -- one image per sampled item
    experiment_1/data/{dataset}_sample.jsonl -- pid, question, answer,
        choices, question_type, answer_type, metadata, image_path
"""

import argparse
import random
from pathlib import Path

from lib.config import DATASET_PRESETS, EXPERIMENT_DIR, load_config, resolve_path
from lib.io_utils import save_jsonl


def resolve_image_field(example: dict, image_fields: tuple[str, ...]):
    """Different benchmarks/dataset versions call the image column
    different things - try each candidate name in `image_fields` (from
    DATASET_PRESETS) until one actually holds a PIL image.
    """
    for field in image_fields:
        val = example.get(field)
        if val is not None and hasattr(val, "save"):  # PIL images have a .save() method
            return val
    raise KeyError(
        f"Could not find a PIL image field on example (tried {image_fields}); "
        f"available keys: {list(example.keys())}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--dataset",
        default=None,
        choices=sorted(DATASET_PRESETS),
        help="Benchmark preset to load. Overrides config's data.dataset.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, dataset=args.dataset)
    data_cfg = cfg["data"]
    dataset_key = data_cfg["dataset"]
    preset = DATASET_PRESETS[dataset_key]

    from datasets import load_dataset  # imported lazily: only needed here, and it's a slow import

    print(f"Loading {preset['hf_id']} split={preset['split']} (preset={dataset_key}) ...")
    ds = load_dataset(preset["hf_id"], split=preset["split"])

    # A fixed seed -> the same n_samples rows every time you re-run this,
    # so re-running data_loading.py doesn't quietly change which examples
    # you're scoring.
    rng = random.Random(data_cfg["seed"])
    n = min(data_cfg["n_samples"], len(ds))
    indices = rng.sample(range(len(ds)), n)

    cache_path = resolve_path(data_cfg["cache_path"])
    image_dir = cache_path.parent / f"images_{dataset_key}"
    image_dir.mkdir(parents=True, exist_ok=True)

    pid_field = preset["pid_field"]
    records = []
    for i in indices:
        example = ds[i]
        pid = str(example.get(pid_field, i)) if pid_field else str(i)
        image = resolve_image_field(example, preset["image_fields"]).convert("RGB")
        image_path = image_dir / f"{pid}.png"
        image.save(image_path)

        records.append(
            {
                "pid": pid,
                "question": example.get(preset["question_field"], ""),
                "answer": example.get(preset["answer_field"], ""),
                "choices": example.get("choices"),
                "question_type": example.get("question_type"),
                "answer_type": example.get("answer_type"),
                "metadata": example.get("metadata"),
                "image_path": str(image_path.relative_to(EXPERIMENT_DIR.parent)),
            }
        )

    save_jsonl(cache_path, records)
    print(f"Saved {len(records)} examples to {cache_path}")
    print(f"Images saved under {image_dir}")


if __name__ == "__main__":
    main()
