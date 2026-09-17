"""Config loading + the dataset-preset registry.

"Config" here means two things layered together:
  1. config.yaml - the knobs you'd actually want to change (which model,
     how many examples, which benchmark, etc).
  2. DATASET_PRESETS below - per-benchmark facts that are NOT knobs (the
     exact Hugging Face dataset id, its split name, and which of its
     columns hold the question/answer/image). These are "facts about the
     dataset", not settings, so they don't live in config.yaml - you pick
     a benchmark by name (data.dataset / --dataset) and this table fills
     in the rest.

Only data_loading.py actually reads DATASET_PRESETS' Hugging Face-specific
fields. Once data_loading.py has cached a benchmark to a local .jsonl file,
every other script (inference.py, decomposition.py) only ever sees that
common pid/question/answer/image_path schema - so which benchmark you're
running never touches the scoring logic itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

# EXPERIMENT_DIR = .../experiment_1  (this file lives at .../experiment_1/lib/config.py)
# REPO_ROOT      = .../prism         (one level up from experiment_1)
# Config paths (cache_path, results_dir, ...) are always written relative to
# REPO_ROOT, so resolve_path() below is what turns them into real paths.
_LIB_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = _LIB_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parent


DATASET_PRESETS = {
    "mathvista": {
        "hf_id": "AI4Math/MathVista",
        "split": "testmini",
        # Recent dataset versions call the image column "decoded_image";
        # older ones call it "image". Try both, in order.
        "image_fields": ("decoded_image", "image"),
        "question_field": "question",
        "answer_field": "answer",
        "pid_field": "pid",
    },
    "hallusionbench": {
        "hf_id": "lmms-lab/HallusionBench",
        "split": "image",  # the visual-question split; excludes the text-only "non_image" split
        "image_fields": ("image",),
        "question_field": "question",
        "answer_field": "gt_answer",
        "pid_field": None,  # question_id is not unique per row; fall back to row index
    },
    "chartqa": {
        "hf_id": "lmms-lab/ChartQA",
        "split": "test",
        "image_fields": ("image",),
        "question_field": "question",
        "answer_field": "answer",
        "pid_field": None,  # no natural unique id; fall back to row index
    },
    "mmmu": {
        "hf_id": "lmms-lab/MMMU",
        "split": "validation",  # "test" has no public ground-truth answers; "dev" is tiny (150)
        # MMMU questions carry up to 7 images (image_1..image_7); most are
        # single-image (image_1) - try each in order, first non-empty wins.
        "image_fields": tuple(f"image_{i}" for i in range(1, 8)),
        "question_field": "question",
        "answer_field": "answer",
        "pid_field": "id",
    },
    "realworldqa": {
        "hf_id": "lmms-lab/RealWorldQA",
        "split": "test",
        "image_fields": ("image",),
        "question_field": "question",
        "answer_field": "answer",
        "pid_field": None,  # no natural unique id; fall back to row index
    },
}


def load_config(config_path: Optional[str] = None, dataset: Optional[str] = None) -> dict:
    """Load config.yaml as a plain dict, with one piece of magic: paths in
    the `data` and `output` sections are templated with "{dataset}" (e.g.
    "experiment_1/results_{dataset}/per_example"), and this function fills
    that placeholder in from `dataset` (an explicit --dataset flag) or,
    if not given, from whatever config.yaml's data.dataset already says.

    This is *the* mechanism that keeps each benchmark's cache/results in
    its own directory without every script needing to know that - they
    just read cfg["output"]["raw_dir"] etc. and it's already the right
    per-benchmark path.
    """
    path = Path(config_path) if config_path else EXPERIMENT_DIR / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if dataset:
        cfg["data"]["dataset"] = dataset

    dataset_key = cfg["data"]["dataset"]
    if dataset_key not in DATASET_PRESETS:
        raise ValueError(
            f"Unknown data.dataset {dataset_key!r}; expected one of {sorted(DATASET_PRESETS)}"
        )

    cfg["data"]["cache_path"] = cfg["data"]["cache_path"].format(dataset=dataset_key)
    for out_key in ("results_dir", "raw_dir", "aggregate_dir"):
        cfg["output"][out_key] = cfg["output"][out_key].format(dataset=dataset_key)

    return cfg


def resolve_path(path_str: str) -> Path:
    """Config paths are written relative to the repo root (e.g.
    "experiment_1/data/..."), not to whatever directory a script happens to
    be run from. This turns such a string into an absolute Path.
    """
    p = Path(path_str)
    return p if p.is_absolute() else REPO_ROOT / p
