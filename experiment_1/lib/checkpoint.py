"""Crash-safe checkpointing: zip a benchmark's results directory (plus
everything needed to make sense of it later) to checkpoints/ so completed
work survives a Kaggle session dying mid-run. Safe to call at any time -
zips whatever exists on disk, including a partially-written per_example/.

Everything is namespaced by model (the --model key, e.g. "qwen3vl"):
results live in results/{model}/{dataset}/, zips in checkpoints/{model}/, and
logs in {LOG_DIR}/{model}/, so runs of different models never overwrite each other.

Besides results/{model}/{dataset}/, each zip captures:
  - the *effective* config used for this run (config.yaml plus any --dataset /
    --model override), since the results alone don't say which model/settings
    produced them
  - meta.json: model key + HF id, timestamp, git commit (+dirty flag), python version, key
    package versions, GPU info - "what code and environment made this"
  - the cached data sample (pid/question/answer/image_path .jsonl) this run
    scored against, so results can be traced back to exact inputs
  - run_status.json, if present (orchestrator progress across benchmarks)
  - this benchmark's GPU log files ({model}/gpu{i}_{dataset}.log),
    if present, for post-mortem debugging of a failed/partial run

Checkpoints go to /kaggle/working/checkpoints/ (persistent output storage)
whenever /kaggle/working exists, regardless of where the repo itself is
checked out; otherwise they fall back to experiment_1/checkpoints/ next to
the repo for local runs.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Optional
from zipfile import ZIP_DEFLATED, ZipFile

from .config import EXPERIMENT_DIR, REPO_ROOT, load_config, resolve_path

# Where run_benchmark.py writes its per-GPU inference logs on Kaggle,
# and where checkpoints themselves land when running on Kaggle.
_KAGGLE_WORKING = Path("/kaggle/working")
LOG_DIR = _KAGGLE_WORKING if _KAGGLE_WORKING.is_dir() else EXPERIMENT_DIR / "logs"

# Packages worth pinning down exactly, since they most affect reproducibility
# of model outputs (versions not installed are silently skipped).
_TRACKED_PACKAGES = [
    "torch", "transformers", "accelerate", "qwen-vl-utils", "numpy",
    "pillow", "pyyaml", "datasets", "huggingface-hub",
]


def checkpoint_dir() -> Path:
    # On Kaggle, force checkpoints into /kaggle/working/checkpoints/ so
    # they're covered by persistent output storage regardless of where the
    # repo itself is checked out. Elsewhere, keep them next to the repo.
    d = _KAGGLE_WORKING / "checkpoints" if _KAGGLE_WORKING.is_dir() else EXPERIMENT_DIR / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_dir(model_key: str) -> Path:
    """Per-model directory for run_benchmark.py's gpu*/decomposition logs."""
    d = LOG_DIR / model_key
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run(cmd: list[str], cwd: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=10, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def _git_info() -> dict:
    return {
        "commit": _run(["git", "rev-parse", "HEAD"], REPO_ROOT),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], REPO_ROOT),
        "dirty": bool(_run(["git", "status", "--porcelain"], REPO_ROOT)),
    }


def _package_versions() -> dict:
    versions = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            pass
    return versions


def _gpu_info() -> list:
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    return [
        {
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "total_memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2),
        }
        for i in range(torch.cuda.device_count())
    ]


def _build_meta(cfg: dict, dataset_key: str, label: str) -> dict:
    return {
        "model": cfg["model"]["key"],
        "model_name": cfg["model"]["name"],
        "dataset": dataset_key,
        "label": label or None,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git": _git_info(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": _package_versions(),
        "gpus": _gpu_info(),
    }


def zip_dataset_results(cfg: dict, dataset_key: str, label: str = "") -> Optional[Path]:
    """Zip results/{model}/{dataset_key}/ (per_example/ and/or aggregate/,
    whichever exist), plus the config/environment/data context needed to
    interpret them, into checkpoints/{model}/{dataset_key}[_{label}]_{timestamp}.zip.
    Returns the zip path, or None if there's nothing to checkpoint yet.
    """
    results_dir = resolve_path(cfg["output"]["results_dir"])
    if not results_dir.exists() or not any(results_dir.rglob("*.*")):
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{label}" if label else ""
    model_key = cfg["model"]["key"]
    zip_dir = checkpoint_dir() / model_key
    zip_dir.mkdir(parents=True, exist_ok=True)
    zip_path = zip_dir / f"{dataset_key}{suffix}_{timestamp}.zip"

    # Write to a temp file, verify, then atomically rename: a kill mid-write
    # (or two shard processes checkpointing in the same second) can never
    # leave a truncated/corrupt zip under the final name.
    tmp_path = zip_path.with_name(f"{zip_path.name}.{os.getpid()}.tmp")
    try:
        _write_zip(tmp_path, cfg, results_dir, dataset_key, label)
        with ZipFile(tmp_path) as zf:
            bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"checkpoint zip failed verification (corrupt member: {bad})")
        os.replace(tmp_path, zip_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return zip_path


def _write_zip(zip_path: Path, cfg: dict, results_dir: Path, dataset_key: str, label: str) -> None:
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as zf:
        for f in results_dir.rglob("*"):
            if f.is_file():
                zf.write(f, arcname=str(f.relative_to(results_dir)))

        zf.writestr("run_config.json", json.dumps(cfg, indent=2))
        zf.writestr("meta.json", json.dumps(_build_meta(cfg, dataset_key, label), indent=2))

        data_cache = resolve_path(cfg["data"]["cache_path"])
        if data_cache.exists():
            zf.write(data_cache, arcname=f"data/{data_cache.name}")

        status_path = checkpoint_dir() / "run_status.json"
        if status_path.exists():
            zf.write(status_path, arcname="run_status.json")

        for pattern in (f"gpu*_{dataset_key}.log", f"decomposition_{dataset_key}.log"):
            for log_path in sorted(log_dir(cfg["model"]["key"]).glob(pattern)):
                zf.write(log_path, arcname=f"logs/{log_path.name}")


def zip_all_configured_datasets(
    config_path: Optional[str], dataset_keys: list[str], model: str
) -> list[Path]:
    """Manual 'zip everything now' entry point: zips current results of
    `model` for each of the given dataset keys, regardless of run progress. Used by
    checkpoint_now.py.
    """
    zipped = []
    for key in dataset_keys:
        cfg = load_config(config_path, dataset=key, model=model)
        path = zip_dataset_results(cfg, key, label="manual")
        if path:
            zipped.append(path)
    return zipped
