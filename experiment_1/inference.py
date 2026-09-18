"""STAGE 2 of the pipeline: the four-condition scoring harness (proposal
§5.1, Table in §5.1) on whichever benchmark's cached sample data_loading.py
(STAGE 1) produced.

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

The actual generation/scoring math lives in lib/scoring.py; this file is
the loop that drives it over a dataset and writes results to disk.

Saves one JSON file per example to experiment_1/results_{dataset}/per_example/.

Usage:
    python experiment_1/inference.py [--config experiment_1/config.yaml] [--dataset ...] [--limit N] [--device cuda:0] [--start N] [--end N]
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

from PIL import Image
from tqdm import tqdm

from lib.checkpoint import zip_dataset_results
from lib.config import DATASET_PRESETS, load_config, resolve_path
from lib.io_utils import load_jsonl
from lib.model import load_model
from lib.progress import clear_progress, report_stage
from lib.scoring import (
    ablate_image,
    cumulative_text,
    generate_blind_trajectory,
    generate_trajectory,
    score_continuation,
    segment_steps,
)

# Checkpoint (zip current results) every N completed examples or every T
# seconds within a shard, whichever comes first - see lib/checkpoint.py.
# This is what makes a Kaggle session dying mid-benchmark non-catastrophic.
CHECKPOINT_EVERY_N_EXAMPLES = 10
CHECKPOINT_EVERY_SECONDS = 600


def scrub_prefix(blind_steps: list[str], s: int) -> str:
    """scrub(y_<s): first (s-1) steps of the BLIND trajectory (the one
    generated with no image), standing in for "a prefix that carries no
    real visual information". s is 1-indexed (step 1's scrub prefix is 0
    steps, i.e. empty). Capped at however many blind steps actually exist,
    in case the blind trajectory turned out shorter than s-1.
    """
    n_take = min(s - 1, len(blind_steps))
    return cumulative_text(blind_steps, n_take)


def run_example(
    loaded, image_full: Image.Image, image_ablated: Image.Image, question: str, cfg: dict, pid: str = ""
) -> dict:
    """Run one example through the full C1-C4 harness: generate the real
    and blind trajectories, then score every step of the real trajectory
    under all four (image, prefix) conditions. Returns everything needed
    to compute M_s/T_s/C_s/S_s/R_s later in decomposition.py (STAGE 3).
    """
    seg_cfg = cfg["segmentation"]

    # The "real" trajectory: the model's actual step-by-step answer, with
    # the real image. This is what we're measuring image-dependence FOR.
    real_text = generate_trajectory(loaded, image_full, question, cfg, seed_tag=f"{pid}:real")
    real_steps = segment_steps(real_text, seg_cfg["method"], seg_cfg["min_chars_per_step"])

    # The "blind" trajectory: a second, independent answer generated with
    # the image ablated from the start. Only used to build scrub(y_<s)
    # below - not scored or reported on its own.
    blind_text = generate_blind_trajectory(loaded, image_ablated, question, cfg, seed_tag=f"{pid}:blind")
    blind_steps = segment_steps(blind_text, seg_cfg["method"], seg_cfg["min_chars_per_step"])

    step_records = []
    n_steps = len(real_steps)
    for s in range(1, n_steps + 1):
        target = real_steps[s - 1]  # the step we're scoring, y_s
        prefix_real = cumulative_text(real_steps, s - 1)  # y_<s : what the model actually said before this step
        prefix_scrub = scrub_prefix(blind_steps, s)  # scrub(y_<s) : a same-length prefix with no real visual info

        # Same target step, four different (image, prefix) combinations.
        # report_stage before each so a stuck/slow forward pass shows up in
        # the monitor as "step s/n_steps C<k>" rather than the whole example
        # just looking frozen with no indication of where.
        report_stage(loaded.device, pid, f"score step {s}/{n_steps} C1")
        ell_1 = score_continuation(loaded, image_full, question, prefix_real, target)  # C1: real image,  real prefix
        report_stage(loaded.device, pid, f"score step {s}/{n_steps} C2")
        ell_2 = score_continuation(loaded, image_ablated, question, prefix_real, target)  # C2: no image,   real prefix
        report_stage(loaded.device, pid, f"score step {s}/{n_steps} C3")
        ell_3 = score_continuation(loaded, image_full, question, prefix_scrub, target)  # C3: real image,  scrubbed prefix
        report_stage(loaded.device, pid, f"score step {s}/{n_steps} C4")
        ell_4 = score_continuation(loaded, image_ablated, question, prefix_scrub, target)  # C4: no image,   scrubbed prefix

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
            result = run_example(loaded, image_full, image_ablated, ex["question"], cfg, pid=ex["pid"])
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

    clear_progress(loaded.device)  # shard done - drop this device's "in flight" entry from the monitor

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
