"""Loading the vision-language model (VLM) itself.

Everything downstream (generation, scoring) is written against the
`LoadedModel` bundle this returns, not against `transformers` directly, so
swapping model families only ever means editing load_model().
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


def load_model(cfg: dict, device_override: Optional[str] = None) -> LoadedModel:
    """Download (if needed) and load the model named in cfg["model"]["name"].

    `device_override` lets a caller pin this to a specific GPU (e.g.
    "cuda:0" vs "cuda:1" for the dual-T4 Kaggle setup) without editing
    config.yaml - see run_multi_shard.py / run_all_benchmarks.py.
    """
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    model_cfg = cfg["model"]
    model_name = model_cfg["name"]
    processor_name = model_cfg.get("processor_name") or model_name

    device = device_override or model_cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(processor_name)

    if "Qwen2-VL" in model_name or "Qwen2.5-VL" in model_name:
        model_cls = Qwen2VLForConditionalGeneration
        if "Qwen2.5-VL" in model_name:
            from transformers import Qwen2_5_VLForConditionalGeneration

            model_cls = Qwen2_5_VLForConditionalGeneration
    else:
        # Generic fallback for other VLM families (e.g. LLaVA) that expose
        # the same AutoModelForVision2Seq-style interface.
        from transformers import AutoModelForVision2Seq

        model_cls = AutoModelForVision2Seq

    model = model_cls.from_pretrained(
        model_name,
        dtype=model_cfg.get("dtype", "auto"),
        device_map=device,
    )
    model.eval()  # inference-only: disables dropout etc, doesn't affect scoring math
    return LoadedModel(model=model, processor=processor, device=device)
