"""The actual experiment logic: turning a (model, image, question) into a
reasoning trajectory, splitting that trajectory into steps, and measuring
how much the model's step-by-step reasoning depends on the image via
teacher-forced log-likelihood scoring.

This is the one file where the scoring MATH lives. See README.md for the
plain-language explanation of what ell_1..ell_4 and the four conditions
(C1-C4) mean; this file is the literal implementation of that table.
"""

from __future__ import annotations

import re

import torch
from PIL import Image

from .model import LoadedModel

# ---------------------------------------------------------------------------
# Image ablation ("blind-drop" condition, proposal §5.4)
#
# "Ablation" = deliberately damaging/removing part of the input to see how
# much the model's output depends on it. Here we replace the real image
# with a featureless one ("Ĩ" in the README/proposal) so the model still
# gets *an* image-shaped input (same size, same number of vision tokens)
# but with none of the actual visual content.
# ---------------------------------------------------------------------------


def ablate_image(image: Image.Image, operator: str = "mean_pixel_fill") -> Image.Image:
    """Return Ĩ, the ablated ("blinded") image. Keeps the original size so
    the number of vision placeholder tokens the processor inserts is
    unchanged between conditions - required for teacher-forced scoring
    (score_continuation below) to line up token-for-token between the
    image-present and image-ablated runs.
    """
    image = image.convert("RGB")
    if operator == "mean_pixel_fill":
        # Simplest ablation: flatten the whole image to its own average
        # color. Removes all visual content but keeps size/exposure similar.
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
# Step segmentation: chopping the model's free-text answer into a list of
# "steps" (y_1, y_2, ..., y_S) so later code can score dependence on the
# image one step at a time, rather than only for the answer as a whole.
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n+")


def segment_steps(text: str, method: str = "sentence", min_chars: int = 4) -> list[str]:
    """Split generated text into a list of "reasoning steps". `method`
    "sentence" splits on sentence boundaries (". ", "? ", "\\n"); "newline"
    splits only on line breaks. Steps shorter than `min_chars` are dropped
    as noise (e.g. a stray "." left over from splitting).
    """
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
    return steps if steps else [text]  # never return zero steps for non-empty text


# ---------------------------------------------------------------------------
# Chat formatting: turning (question, optional partial answer) into the
# model's expected chat-template input.
# ---------------------------------------------------------------------------

# This exact instruction is used for BOTH generating trajectories
# (generate_trajectory / generate_blind_trajectory) AND teacher-forced
# scoring (score_continuation), so ell_1..ell_4 are always computed under
# the exact prompt that produced the trajectory being scored - a mismatch
# here would bias the log-likelihoods against whatever prompt wording
# actually generated the step under evaluation.
STEP_BY_STEP_INSTRUCTION = (
    "Solve this step by step, showing your reasoning for each step, "
    "then give your final answer."
)


def build_messages(question: str) -> list[dict]:
    """Build a chat-template message list holding just the one user turn
    (image + question). Deliberately never includes an assistant turn - see
    build_prompt_prefix()'s docstring for why priming with a partial answer
    is done by raw string concatenation instead of a second chat message.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": f"{question}\n{STEP_BY_STEP_INSTRUCTION}",
                },
            ],
        }
    ]


def build_prompt_prefix(processor, question: str) -> str:
    """Render the chat template through the end of the user turn plus the
    assistant-turn preamble (add_generation_prompt=True) - i.e. exactly the
    text real generation is conditioned on before the model writes its
    first token. Any continuation (a real or scrub prefix, or a scored
    target step) is appended to this as a raw string, never as a second
    chat message.

    This matters specifically for "Thinking"-style models (e.g.
    Qwen3-VL-*-Thinking): their chat template renders
    add_generation_prompt=True as "...<|im_start|>assistant\\n<think>\\n",
    so generation always starts inside an open <think> block. If a partial
    trajectory were instead passed back in as a second {"role": "assistant",
    "content": ...} message (as this pipeline used to do), that template
    re-parses the content for "<think>...</think>" and re-wraps it with a
    *synthetic closing </think>* even when the real generation never closed
    the block at that point (because it's a mid-reasoning prefix, not a
    finished turn) - silently feeding the model a different context than
    the one that actually produced the text being scored. Building the
    prompt as one flat string sidesteps that template logic entirely, so
    teacher-forced scoring always sees the exact token stream real
    generation would have produced up to that point.
    """
    messages = build_messages(question)
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------------------
# Generation: asking the model to actually produce a reasoning trajectory.
# ---------------------------------------------------------------------------


def extract_trajectory_text(raw_text: str) -> str:
    """Turn a model's raw generated text into the flat trajectory text used
    for step segmentation and scoring.

    A "Thinking"-style model's real output is "{reasoning}</think>\\n\\n
    {final answer}" - generation starts already inside an open <think>
    block because add_generation_prompt renders "...<think>\\n" (see
    build_prompt_prefix()). Per this pilot's design decision, the
    trajectory being measured for visual grounding is the *whole* thing -
    reasoning and final answer concatenated - not just one half; this
    drops only the structural "</think>" marker itself, joining the two
    parts with a space so segment_steps() sees one continuous text.

    Non-thinking models (e.g. Qwen2-VL) never emit "</think>", so this is a
    no-op for them - raw_text is returned unchanged (stripped).
    """
    if "</think>" in raw_text:
        reasoning, _, answer = raw_text.partition("</think>")
        return (reasoning.strip() + " " + answer.strip()).strip()
    return raw_text.strip()


def _seed_for(cfg: dict, seed_tag: str) -> int:
    """Deterministic seed derived from config's model.generation_seed plus
    a per-call tag (e.g. "{pid}:real" / "{pid}:blind"), so sampling-based
    generation is still reproducible across re-runs without every example
    sharing the exact same random draw (which reusing one global seed
    outright would cause).
    """
    import hashlib

    base = int(cfg["model"].get("generation_seed", 42))
    digest = hashlib.sha256(f"{base}:{seed_tag}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def generate_trajectory(
    loaded: LoadedModel, image: Image.Image, question: str, cfg: dict, seed_tag: str = ""
) -> str:
    """Generate the model's free-text step-by-step answer to `question`
    about `image`. This is a normal autoregressive generation call (the
    model picks its own tokens); teacher-forced *scoring* of that text
    against alternative conditions happens separately, in
    score_continuation() below.
    """
    model, processor, device = loaded.model, loaded.processor, loaded.device
    model_cfg = cfg["model"]

    text = build_prompt_prefix(processor, question)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)

    gen_kwargs = dict(max_new_tokens=model_cfg.get("max_new_tokens", 256))
    if model_cfg.get("do_sample", True):
        # Qwen3-VL's model card: greedy decoding in "Thinking" mode can
        # cause performance degradation and endless repetition loops, so
        # sampling is the default here (not greedy) - see config.yaml.
        gen_kwargs.update(
            do_sample=True,
            temperature=model_cfg.get("temperature", 0.6),
            top_p=model_cfg.get("top_p", 0.95),
            top_k=model_cfg.get("top_k", 20),
        )
    else:
        gen_kwargs.update(do_sample=False)

    if seed_tag:
        torch.manual_seed(_seed_for(cfg, seed_tag))

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    new_ids = output_ids[:, prompt_len:]  # drop the prompt tokens, keep only what the model generated
    text_out = processor.batch_decode(new_ids, skip_special_tokens=True)[0]
    return extract_trajectory_text(text_out)


def generate_blind_trajectory(
    loaded: LoadedModel, ablated_image: Image.Image, question: str, cfg: dict, seed_tag: str = ""
) -> str:
    """The "scrub source": a trajectory (z_1, ..., z_Z) generated by the
    SAME model asked the SAME question, but with the image ablated from the
    very start (proposal §5.3, "blind regeneration"). Used to build
    scrub(y_<s) - a stand-in prefix that carries no leftover visual
    information, needed for the C3/C4 conditions. See inference.py's
    scrub_prefix() for how this trajectory gets turned into that prefix.
    """
    return generate_trajectory(loaded, ablated_image, question, cfg, seed_tag=seed_tag)


# ---------------------------------------------------------------------------
# Teacher-forced scoring: ell = (1/|y_s|) * log pi(y_s | image, prefix)
#
# "Teacher-forced" means: instead of letting the model generate freely, we
# FEED it a specific piece of text (the "target") and ask "how likely did
# you think each of these tokens were, given everything before them?" -
# i.e. we score text the model already produced (or a stand-in for it)
# rather than sampling new text. This is what lets us compare the same
# step y_s under four different (image, prefix) conditions.
# ---------------------------------------------------------------------------


def _longest_common_prefix_len(a: torch.Tensor, b: torch.Tensor) -> int:
    """How many leading tokens two token-id sequences share. Used below to
    find exactly where "prefix" ends and "target" begins once both have
    been tokenized - see score_continuation()'s docstring for why this is
    needed instead of just knowing len(prefix_tokens).
    """
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
    This is a single forward pass (no generation) - we already know the
    text, we're just asking how probable the model thinks it was.

    Tokenizes the prefix-only text and the prefix+target text separately
    and uses their longest common token prefix as the split point. This is
    more robust than just using len(prefix_tokens) directly, because
    tokenizers can re-split characters differently right at the seam where
    prefix meets target (a BPE/subword-tokenizer quirk) - the standard trick
    used by continuation-scoring harnesses.

    prefix_text/target_text are appended as raw strings onto
    build_prompt_prefix()'s output, never as a second chat message - see
    that function's docstring for why (Thinking-model chat templates would
    otherwise re-parse and corrupt a mid-reasoning prefix).
    """
    model, processor, device = loaded.model, loaded.processor, loaded.device

    prompt_text = build_prompt_prefix(processor, question)
    text_prefix = prompt_text + prefix_text
    text_full = prompt_text + prefix_text + target_text

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

    # logits[t-1] predicts token at position t, so to score the target
    # tokens (positions split..end) we need the logits one position earlier.
    target_positions = torch.arange(split, full_ids_dev.shape[0], device=logits.device)
    pred_logits = logits[target_positions - 1]
    target_ids = full_ids_dev[target_positions]

    log_probs = torch.log_softmax(pred_logits.float(), dim=-1)
    token_logps = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)

    return float(token_logps.mean().item())  # mean, not sum: length-normalised so steps are comparable


def cumulative_text(steps: list[str], upto: int) -> str:
    """Join steps[0:upto] back into a single prefix string, matching the
    whitespace the model would have produced (one space between steps).
    """
    return " ".join(steps[:upto]).strip()
