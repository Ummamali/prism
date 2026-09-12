"""Helper functions for running Qwen3-VL inference: CLI parsing, input
preprocessing, and output decoding."""

import argparse
from pathlib import Path

import torch

IMAGES_DIR = Path(__file__).resolve().parent / "images"
DEFAULT_IMAGE = "car.jpg"
DEFAULT_PROMPT = "Describe this image in one sentence."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Qwen3-VL example.")
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help="Image URL, absolute/relative path, or filename under images/ (e.g. car.jpg)",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Text prompt")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def resolve_image(image: str) -> str:
    """Resolve a bare filename (e.g. "car.jpg") to images/<file>.

    URLs and paths that already exist as given (absolute or relative to the
    current directory) are returned unchanged.
    """
    if image.startswith(("http://", "https://")) or Path(image).exists():
        return image
    candidate = IMAGES_DIR / image
    if candidate.exists():
        return str(candidate)
    return image


def build_inputs(processor, image: str, prompt: str, device: str):
    # Qwen3-VL expects the chat-style {role, content:[...]} format so the
    # processor can interleave image and text tokens correctly.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    # apply_chat_template both tokenizes the text and preprocesses the image
    # (fetching it from a URL/path) in one call.
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return inputs.to(device)


def generate_answer(model, processor, inputs, max_new_tokens: int) -> str:
    with torch.no_grad():  # skip gradient tracking, inference only
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    # generate() returns prompt + completion tokens concatenated, so slice
    # off the prompt length to keep only the newly generated response.
    trimmed_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(
        trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
