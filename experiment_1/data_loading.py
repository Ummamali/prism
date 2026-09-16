"""Pull a small pilot sample from AI4Math/MathVista and cache it locally.

Usage:
    python experiment_1/data_loading.py [--config experiment_1/config.yaml]

Saves:
    experiment_1/data/images/{pid}.png   -- one image per sampled item
    experiment_1/data/mathvista_sample.jsonl -- pid, question, answer,
        choices, question_type, answer_type, metadata, image_path
"""

import argparse
import random
from pathlib import Path

from common import EXPERIMENT_DIR, load_config, resolve_path, save_jsonl

# MathVista's HF schema uses "decoded_image" for the PIL image in recent
# dataset versions and "image" (a relative path string) in older ones.
IMAGE_FIELD_CANDIDATES = ("decoded_image", "image")


def resolve_image_field(example: dict):
    for field in IMAGE_FIELD_CANDIDATES:
        val = example.get(field)
        if val is not None and hasattr(val, "save"):
            return val
    raise KeyError(
        f"Could not find a PIL image field on example (tried {IMAGE_FIELD_CANDIDATES}); "
        f"available keys: {list(example.keys())}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]

    from datasets import load_dataset

    print(f"Loading {data_cfg['dataset']} split={data_cfg['split']} ...")
    ds = load_dataset(data_cfg["dataset"], split=data_cfg["split"])

    rng = random.Random(data_cfg["seed"])
    n = min(data_cfg["n_samples"], len(ds))
    indices = rng.sample(range(len(ds)), n)

    cache_path = resolve_path(data_cfg["cache_path"])
    image_dir = cache_path.parent / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i in indices:
        example = ds[i]
        pid = str(example.get("pid", i))
        image = resolve_image_field(example).convert("RGB")
        image_path = image_dir / f"{pid}.png"
        image.save(image_path)

        records.append(
            {
                "pid": pid,
                "question": example.get("question", ""),
                "answer": example.get("answer", ""),
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
