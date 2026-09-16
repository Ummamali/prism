"""Backward-compatibility shim.

`common.py` used to hold everything (config loading, model loading, scoring
math, ...) in one 350-line file. It's since been split up for readability
into experiment_1/lib/ - see README.md's "Codebase map" section for what
lives where. This file just re-exports the old names so anything still
doing `from common import X` (e.g. a monitoring snippet already pasted into
a running Kaggle notebook cell) keeps working without edits.

New code, and anything you're pasting fresh, should import from lib.*
directly instead - e.g. `from lib.config import load_config`.
"""

from lib.config import DATASET_PRESETS, EXPERIMENT_DIR, REPO_ROOT, load_config, resolve_path
from lib.io_utils import load_jsonl, save_jsonl
from lib.model import LoadedModel, load_model
from lib.scoring import (
    STEP_BY_STEP_INSTRUCTION,
    ablate_image,
    build_messages,
    cumulative_text,
    generate_blind_trajectory,
    generate_trajectory,
    score_continuation,
    segment_steps,
)

__all__ = [
    "DATASET_PRESETS",
    "EXPERIMENT_DIR",
    "REPO_ROOT",
    "load_config",
    "resolve_path",
    "load_jsonl",
    "save_jsonl",
    "LoadedModel",
    "load_model",
    "STEP_BY_STEP_INSTRUCTION",
    "ablate_image",
    "build_messages",
    "cumulative_text",
    "generate_blind_trajectory",
    "generate_trajectory",
    "score_continuation",
    "segment_steps",
]
