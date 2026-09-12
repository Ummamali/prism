"""Run a single image-captioning example on Qwen3-VL-2B-Instruct (smallest Qwen3-VL model).

Usage:
    python run_qwen3_vl.py
    python run_qwen3_vl.py --image path/or/url --prompt "Describe this image."
"""

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# build_inputs/generate_answer/parse_args/resolve_image live in utils.py —
# this script is just the high-level orchestration (load model -> run once).
from utils import build_inputs, generate_answer, parse_args, resolve_image

# Qwen3-VL-2B is the smallest dense checkpoint in the Qwen3-VL family
# (2B/4B/8B/30B/32B/235B); larger ones need proportionally more RAM/VRAM.
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"


def load_model(model_id: str):
    # Falls back to CPU automatically when no CUDA GPU is present.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # AutoProcessor bundles the tokenizer (for text) and image processor
    # (for resizing/normalizing images) that this specific model expects.
    processor = AutoProcessor.from_pretrained(model_id)

    # Downloads (and caches locally) the model weights, then loads them.
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype="auto",  # let transformers pick the checkpoint's native dtype (bf16)
        device_map=device,  # places the model weights on "cuda" or "cpu"
    )
    model.eval()  # disable dropout etc. for deterministic inference
    return model, processor, device


def main() -> None:
    # Reads --image / --prompt / --max-new-tokens from the command line
    # (see utils.parse_args for defaults).
    args = parse_args()

    print(f"Loading {MODEL_ID} ...")
    model, processor, device = load_model(MODEL_ID)
    print(f"Model loaded on {device}.")

    # Turn "car.jpg" into a real path under images/, or pass URLs/paths through.
    image = resolve_image(args.image)

    # Formats the (image, prompt) pair into the tensors the model's
    # generate() call expects, and moves them onto the right device.
    inputs = build_inputs(processor, image, args.prompt, device)

    print("Generating response ...")
    # Runs the model forward to produce the text answer.
    answer = generate_answer(model, processor, inputs, args.max_new_tokens)

    print("\nPrompt:", args.prompt)
    print("Image:", image)
    print("Response:", answer)


if __name__ == "__main__":
    main()
