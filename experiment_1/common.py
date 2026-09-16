"""Shared helpers for Experiment 1: config loading, model loading, image
ablation, CoT step segmentation, and teacher-forced log-likelihood scoring.

The four-condition scoring scheme (C1-C4) and the M_s/T_s/C_s/S_s/R_s
definitions implemented here follow research_proposal_prism.pdf §5.1
literally. Two implementation choices are NOT fully specified by the
proposal and were resolved with the user before writing this code:
  - scrub(y_<s) via blind regeneration, matched by step index (not token
    count) against a same-model trajectory generated with the image
    ablated from the start. See generate_blind_trajectory().
  - CoT steps are sentence-split segments of the generated text.
See experiment_1/README.md for the full list of pilot-scope decisions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import yaml
from PIL import Image

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parent


def load_config(config_path: Optional[str] = None) -> dict:
    path = Path(config_path) if config_path else EXPERIMENT_DIR / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_str: str) -> Path:
    """Config paths are written relative to the repo root."""
    p = Path(path_str)
    return p if p.is_absolute() else REPO_ROOT / p


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


@dataclass
class LoadedModel:
    model: object
    processor: object
    device: str


def load_model(cfg: dict) -> LoadedModel:
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    model_cfg = cfg["model"]
    model_name = model_cfg["name"]
    processor_name = model_cfg.get("processor_name") or model_name

    device = model_cfg.get("device", "auto")
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
    model.eval()
    return LoadedModel(model=model, processor=processor, device=device)


# ---------------------------------------------------------------------------
# Image ablation ("blind-drop" condition, proposal §5.4)
# ---------------------------------------------------------------------------


def ablate_image(image: Image.Image, operator: str = "mean_pixel_fill") -> Image.Image:
    """Return Ĩ, the ablated image. Keeps the original size so the number
    of vision placeholder tokens the processor inserts is unchanged between
    conditions, which is required for teacher-forced scoring to line up.
    """
    image = image.convert("RGB")
    if operator == "mean_pixel_fill":
        import numpy as np

        arr = np.asarray(image)
        mean_color = tuple(int(c) for c in arr.reshape(-1, 3).mean(axis=0))
        return Image.new("RGB", image.size, mean_color)
    elif operator == "gaussian_noise":
        import numpy as np

        arr = np.asarray(image).astype("float32")
        mean, std = arr.mean(), arr.std() + 1e-6
        noise = np.random.default_rng(0).normal(mean, std, arr.shape)
        noise = noise.clip(0, 255).astype("uint8")
        return Image.fromarray(noise)
    else:
        raise ValueError(f"Unknown ablation operator: {operator}")


# ---------------------------------------------------------------------------
# Step segmentation
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n+")


def segment_steps(text: str, method: str = "sentence", min_chars: int = 4) -> list[str]:
    text = text.strip()
    if not text:
        return []

    if method == "sentence":
        parts = _SENTENCE_SPLIT_RE.split(text)
    elif method == "newline":
        parts = text.split("\n")
    else:
        raise ValueError(f"Unknown segmentation method: {method}")

    steps = [p.strip() for p in parts if len(p.strip()) >= min_chars]
    return steps if steps else [text]


# ---------------------------------------------------------------------------
# Chat formatting
# ---------------------------------------------------------------------------


def build_messages(question: str, assistant_text: Optional[str] = None) -> list[dict]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        f"{question}\nThink step by step, then give the final answer."
                    ),
                },
            ],
        }
    ]
    if assistant_text is not None:
        messages.append({"role": "assistant", "content": assistant_text})
    return messages


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_trajectory(loaded: LoadedModel, image: Image.Image, question: str, cfg: dict) -> str:
    model, processor, device = loaded.model, loaded.processor, loaded.device
    model_cfg = cfg["model"]

    messages = build_messages(question)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)

    gen_kwargs = dict(max_new_tokens=model_cfg.get("max_new_tokens", 256))
    if model_cfg.get("do_sample", False):
        gen_kwargs.update(do_sample=True, temperature=model_cfg.get("temperature", 1.0))
    else:
        gen_kwargs.update(do_sample=False)

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    new_ids = output_ids[:, prompt_len:]
    text_out = processor.batch_decode(new_ids, skip_special_tokens=True)[0]
    return text_out.strip()


def generate_blind_trajectory(
    loaded: LoadedModel, ablated_image: Image.Image, question: str, cfg: dict
) -> str:
    """Scrub source: a trajectory generated by the same model with the
    image ablated from the start (proposal §5.3, blind regeneration).
    """
    return generate_trajectory(loaded, ablated_image, question, cfg)


# ---------------------------------------------------------------------------
# Teacher-forced scoring: ell = (1/|y_s|) * log pi(y_s | image, prefix)
# ---------------------------------------------------------------------------


def _longest_common_prefix_len(a: torch.Tensor, b: torch.Tensor) -> int:
    n = min(a.shape[0], b.shape[0])
    if n == 0:
        return 0
    mismatch = (a[:n] != b[:n]).nonzero(as_tuple=True)[0]
    return int(mismatch[0]) if len(mismatch) else n


def score_continuation(
    loaded: LoadedModel,
    image: Image.Image,
    question: str,
    prefix_text: str,
    target_text: str,
) -> float:
    """Length-normalised log-likelihood (nats/token) of target_text given
    (image, question, prefix_text), i.e. ell = (1/|y_s|) log pi(y_s | ...).

    Tokenizes the prefix-only text and the prefix+target text separately
    and uses their longest common token prefix as the split point, which
    is robust to BPE boundary re-tokenization at the prefix/target seam
    (the standard trick used by continuation-scoring harnesses).
    """
    model, processor, device = loaded.model, loaded.processor, loaded.device

    messages_prefix = build_messages(question, assistant_text=prefix_text)
    messages_full = build_messages(question, assistant_text=prefix_text + target_text)

    text_prefix = processor.apply_chat_template(
        messages_prefix, tokenize=False, add_generation_prompt=False
    )
    text_full = processor.apply_chat_template(
        messages_full, tokenize=False, add_generation_prompt=False
    )

    inputs_prefix = processor(text=[text_prefix], images=[image], return_tensors="pt", padding=True)
    inputs_full = processor(text=[text_full], images=[image], return_tensors="pt", padding=True).to(
        device
    )

    prefix_ids_cpu = inputs_prefix["input_ids"][0]
    full_ids_dev = inputs_full["input_ids"][0]  # stays on `device`
    split = _longest_common_prefix_len(prefix_ids_cpu, full_ids_dev.to(prefix_ids_cpu.device))

    if split >= full_ids_dev.shape[0]:
        # Degenerate: target contributed no new tokens (e.g. empty step).
        return 0.0

    with torch.no_grad():
        outputs = model(**inputs_full)
    logits = outputs.logits[0]  # [seq_len, vocab]

    # logits[t-1] predicts token at position t.
    target_positions = torch.arange(split, full_ids_dev.shape[0], device=logits.device)
    pred_logits = logits[target_positions - 1]
    target_ids = full_ids_dev[target_positions]

    log_probs = torch.log_softmax(pred_logits.float(), dim=-1)
    token_logps = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)

    return float(token_logps.mean().item())


def cumulative_text(steps: list[str], upto: int) -> str:
    """Concatenate steps[0:upto] into a prefix string, matching the
    whitespace the model would have produced (one space between steps).
    """
    return " ".join(steps[:upto]).strip()


def save_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
