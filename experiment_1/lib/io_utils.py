"""Tiny helpers for reading/writing the .jsonl files this pipeline passes
between stages (one JSON object per line - "JSON Lines", a common format
for datasets because you can append to it or read it line-by-line without
loading the whole file into memory).
"""

from __future__ import annotations

import json
from pathlib import Path


def save_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
