"""Pull a small pilot sample from the configured benchmark (data.dataset in
config.yaml, or --dataset) and cache it locally.

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

from common import DATASET_PRESETS, EXPERIMENT_DIR, load_config, resolve_path, save_jsonl


def resolve_image_field(example: dict, image_fields: tuple[str, ...]):
    for field in image_fields:
        val = example.get(field)
        if val is not None and hasattr(val, "save"):
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

    from datasets import load_dataset

    print(f"Loading {preset['hf_id']} split={preset['split']} (preset={dataset_key}) ...")
    ds = load_dataset(preset["hf_id"], split=preset["split"])

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
