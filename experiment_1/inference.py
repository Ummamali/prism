"""Run the four-condition scoring harness (proposal §5.1, Table in §5.1)
on the cached MathVista pilot sample.

For each example:
  1. Generate the real trajectory y = (y_1..y_S) with the image present.
  2. Generate a blind trajectory z = (z_1..z_Z) with the image ablated
     from the start (the scrub source, §5.3 blind regeneration).
  3. For each step s, teacher-force-score y_s under four conditions:
       C1: image=I,  prefix=y_<s          -> ell_1
       C2: image=Ĩ,  prefix=y_<s          -> ell_2
       C3: image=I,  prefix=scrub(y_<s)   -> ell_3
       C4: image=Ĩ,  prefix=scrub(y_<s)   -> ell_4
     where scrub(y_<s) = z_<s (first s-1 steps of the blind trajectory,
     capped at len(z) if the blind trajectory is shorter), matched by
     step index rather than token count -- see README for why.

Saves one JSON file per example to experiment_1/results/per_example/.

Usage:
    python experiment_1/inference.py [--config experiment_1/config.yaml] [--limit N]
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

from PIL import Image
from tqdm import tqdm

from checkpoint import zip_dataset_results
from common import (
    DATASET_PRESETS,
    ablate_image,
    cumulative_text,
    generate_blind_trajectory,
    generate_trajectory,
    load_config,
    load_jsonl,
    load_model,
    resolve_path,
    score_continuation,
    segment_steps,
)

# Checkpoint (zip current results) every N completed examples or every T
# seconds within a shard, whichever comes first - see checkpoint.py.
CHECKPOINT_EVERY_N_EXAMPLES = 10
CHECKPOINT_EVERY_SECONDS = 600


def scrub_prefix(blind_steps: list[str], s: int) -> str:
    """scrub(y_<s): first (s-1) steps of the blind trajectory, capped at
    however many blind steps exist. s is 1-indexed.
    """
    n_take = min(s - 1, len(blind_steps))
    return cumulative_text(blind_steps, n_take)


def run_example(loaded, image_full: Image.Image, image_ablated: Image.Image, question: str, cfg: dict) -> dict:
    seg_cfg = cfg["segmentation"]

    real_text = generate_trajectory(loaded, image_full, question, cfg)
    real_steps = segment_steps(real_text, seg_cfg["method"], seg_cfg["min_chars_per_step"])

    blind_text = generate_blind_trajectory(loaded, image_ablated, question, cfg)
    blind_steps = segment_steps(blind_text, seg_cfg["method"], seg_cfg["min_chars_per_step"])

    step_records = []
    for s in range(1, len(real_steps) + 1):
        target = real_steps[s - 1]
        prefix_real = cumulative_text(real_steps, s - 1)
        prefix_scrub = scrub_prefix(blind_steps, s)

        ell_1 = score_continuation(loaded, image_full, question, prefix_real, target)
        ell_2 = score_continuation(loaded, image_ablated, question, prefix_real, target)
        ell_3 = score_continuation(loaded, image_full, question, prefix_scrub, target)
        ell_4 = score_continuation(loaded, image_ablated, question, prefix_scrub, target)

        step_records.append(
            {
                "step_index": s,
                "step_text": target,
                "ell_1": ell_1,
                "ell_2": ell_2,
                "ell_3": ell_3,
                "ell_4": ell_4,
            }
        )

    return {
        "real_trajectory": real_text,
        "real_steps": real_steps,
        "blind_trajectory": blind_text,
        "blind_steps": blind_steps,
        "n_steps": len(real_steps),
        "steps": step_records,
    }


def run_shard(
    loaded,
    cfg: dict,
    dataset_key: str,
    start: Optional[int] = None,
    end: Optional[int] = None,
    limit: Optional[int] = None,
) -> None:
    """Score one slice [start:end] of `dataset_key`'s cached sample with an
    already-loaded model, checkpointing (zipping) results periodically so a
    crashed/disconnected session doesn't lose completed work. Reusable from
    both `main()` (single-shard CLI) and run_multi_shard.py (one process
    looping over several benchmark shards without reloading the model).
    """
    data_cfg = cfg["data"]
    out_cfg = cfg["output"]
    ablation_op = cfg["ablation"]["image_operator"]

    cache_path = resolve_path(data_cfg["cache_path"])
    examples = load_jsonl(cache_path)
    examples = examples[start:end]
    if limit:
        examples = examples[:limit]

    raw_dir = resolve_path(out_cfg["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    n_since_checkpoint = 0
    last_checkpoint_time = time.monotonic()

    for ex in tqdm(examples, desc=f"Scoring {dataset_key}"):
        out_path = raw_dir / f"{ex['pid']}.json"
        if out_path.exists():
            continue  # resume-friendly: skip already-scored examples

        image_path = resolve_path(ex["image_path"])
        image_full = Image.open(image_path).convert("RGB")
        image_ablated = ablate_image(image_full, ablation_op)

        try:
            result = run_example(loaded, image_full, image_ablated, ex["question"], cfg)
        except Exception as e:  # noqa: BLE001 - pilot harness, log and continue
            print(f"[WARN] example {ex['pid']} failed: {e}")
            continue

        result["pid"] = ex["pid"]
        result["question"] = ex["question"]
        result["answer"] = ex["answer"]

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        n_since_checkpoint += 1
        elapsed = time.monotonic() - last_checkpoint_time
        if n_since_checkpoint >= CHECKPOINT_EVERY_N_EXAMPLES or elapsed >= CHECKPOINT_EVERY_SECONDS:
            zip_path = zip_dataset_results(cfg, dataset_key, label="partial")
            if zip_path:
                print(f"[checkpoint] {zip_path}")
            n_since_checkpoint = 0
            last_checkpoint_time = time.monotonic()

    zip_path = zip_dataset_results(cfg, dataset_key, label="shard_done")
    if zip_path:
        print(f"[checkpoint] shard finished -> {zip_path}")

    print(f"Per-example results written to {raw_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--dataset",
        default=None,
        choices=sorted(DATASET_PRESETS),
        help="Benchmark preset to run. Overrides config's data.dataset.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N examples.")
    parser.add_argument(
        "--device",
        default=None,
        help="Device to load the model on, e.g. cuda:0 or cuda:1. Overrides config's model.device.",
    )
    parser.add_argument(
        "--start", type=int, default=None, help="Start index (inclusive) of the example slice to run."
    )
    parser.add_argument(
        "--end", type=int, default=None, help="End index (exclusive) of the example slice to run."
    )
    args = parser.parse_args()

    cfg = load_config(args.config, dataset=args.dataset)

    print(f"Loading model {cfg['model']['name']} ...")
    loaded = load_model(cfg, device_override=args.device)
    print(f"Model loaded on {loaded.device}.")

    run_shard(loaded, cfg, cfg["data"]["dataset"], start=args.start, end=args.end, limit=args.limit)


if __name__ == "__main__":
    main()
