"""Loading the vision-language model (VLM) itself.

Everything downstream (generation, scoring) is written against the
`LoadedModel` bundle this returns, not against `transformers` directly, so
swapping model families only ever means adding a MODEL_PRESETS entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class LoadedModel:
    """Bundles the three things every generation/scoring call needs
    together, so functions can take one `loaded` argument instead of three.
    """

    model: object
    processor: object  # turns (image, text) into model inputs, and ids back into text
    device: str  # e.g. "cuda:0" - where `model`'s weights actually live
    # System prompt prepended to every chat (None = no system turn). Used to
    # switch InternVL3.5 into Thinking Mode; see MODEL_PRESETS.
    system_prompt: Optional[str] = None
    # Raw text appended after the chat template's generation prompt (see
    # MODEL_PRESETS); "" for models whose template already opens <think>.
    generation_prefix: str = ""


def load_model(cfg: dict, device_override: Optional[str] = None) -> LoadedModel:
    """Download (if needed) and load the model selected by --model, i.e. the
    MODEL_PRESETS entry that load_config() copied into cfg["model"]
    (name = HF id, model_class = transformers loader class name).

    `device_override` lets a caller pin this to a specific GPU (e.g.
    "cuda:0" vs "cuda:1" for the dual-T4 Kaggle setup) without editing
    config.yaml - see run_multi_shard.py / run_benchmark.py.
    """
    import transformers
    from transformers import AutoProcessor

    model_cfg = cfg["model"]
    model_name = model_cfg["name"]
    processor_name = model_cfg.get("processor_name") or model_name
    trust_remote_code = model_cfg.get("trust_remote_code", False)

    device = device_override or model_cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(processor_name, trust_remote_code=trust_remote_code)

    model_cls = getattr(transformers, model_cfg["model_class"])
    model = model_cls.from_pretrained(
        model_name,
        dtype=model_cfg.get("dtype", "auto"),
        device_map=device,
        trust_remote_code=trust_remote_code,
    )
    model.eval()  # inference-only: disables dropout etc, doesn't affect scoring math
    return LoadedModel(
        model=model,
        processor=processor,
        device=device,
        system_prompt=model_cfg.get("system_prompt"),
        generation_prefix=model_cfg.get("generation_prefix", ""),
    )
