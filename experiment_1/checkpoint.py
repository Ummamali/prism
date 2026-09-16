"""Crash-safe checkpointing: zip a benchmark's results directory to
experiment_1/checkpoints/ so completed work survives a Kaggle session dying
mid-run. Safe to call at any time - zips whatever exists on disk, including
a partially-written per_example/.

experiment_1/checkpoints/ resolves under the repo root, which on Kaggle is
expected to live under /kaggle/working/ (persistent output storage) - see
README/execution plan. If your repo lives elsewhere on Kaggle, point
--config at a config whose paths still resolve under /kaggle/working/.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from common import EXPERIMENT_DIR, load_config, resolve_path


def checkpoint_dir() -> Path:
    d = EXPERIMENT_DIR / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def zip_dataset_results(cfg: dict, dataset_key: str, label: str = "") -> Optional[Path]:
    """Zip results_{dataset_key}/ (per_example/ and/or aggregate/, whichever
    exist) into experiment_1/checkpoints/{dataset_key}[_{label}]_{timestamp}.zip.
    Returns the zip path, or None if there's nothing to checkpoint yet.
    """
    results_dir = resolve_path(cfg["output"]["results_dir"])
    if not results_dir.exists() or not any(results_dir.rglob("*.*")):
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{label}" if label else ""
    zip_stem = checkpoint_dir() / f"{dataset_key}{suffix}_{timestamp}"

    zip_path = shutil.make_archive(str(zip_stem), "zip", root_dir=results_dir)
    return Path(zip_path)


def zip_all_configured_datasets(config_path: Optional[str], dataset_keys: list[str]) -> list[Path]:
    """Manual 'zip everything now' entry point: zips current results for
    each of the given dataset keys, regardless of run progress. Used by
    checkpoint_now.py.
    """
    zipped = []
    for key in dataset_keys:
        cfg = load_config(config_path, dataset=key)
        path = zip_dataset_results(cfg, key, label="manual")
        if path:
            zipped.append(path)
    return zipped
