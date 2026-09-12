"""Run a single image-captioning example on Qwen3-VL-2B-Instruct (smallest Qwen3-VL model).

Usage:
    python run_qwen3_vl.py
    python run_qwen3_vl.py --image path/or/url --prompt "Describe this image."
"""

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from utils import build_inputs, generate_answer, parse_args, resolve_image

# Qwen3-VL-2B is the smallest dense checkpoint in the Qwen3-VL family
# (2B/4B/8B/30B/32B/235B); larger ones need proportionally more RAM/VRAM.
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"


def load_model(model_id: str):
    # Falls back to CPU automatically when no CUDA GPU is present.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype="auto",  # let transformers pick the checkpoint's native dtype (bf16)
        device_map=device,
    )
    model.eval()  # disable dropout etc. for deterministic inference
    return model, processor, device


def main() -> None:
    args = parse_args()

    print(f"Loading {MODEL_ID} ...")
    model, processor, device = load_model(MODEL_ID)
    print(f"Model loaded on {device}.")

    image = resolve_image(args.image)
    inputs = build_inputs(processor, image, args.prompt, device)

    print("Generating response ...")
    answer = generate_answer(model, processor, inputs, args.max_new_tokens)

    print("\nPrompt:", args.prompt)
    print("Image:", image)
    print("Response:", answer)


if __name__ == "__main__":
    main()
