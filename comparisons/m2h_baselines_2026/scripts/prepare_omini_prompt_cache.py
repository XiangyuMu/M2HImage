#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast


DEFAULT_PROMPT = (
    "Transform the mannequin into the referenced person while preserving the outfit, "
    "pose, and background."
)
FORMAT_VERSION = "1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute the two fixed FLUX prompt embeddings used by M2H OminiControl."
    )
    parser.add_argument("--flux-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def metadata(args: argparse.Namespace) -> dict[str, str]:
    return {
        "format_version": FORMAT_VERSION,
        "flux_path": str(Path(args.flux_path).resolve()),
        "prompt": args.prompt,
        "max_sequence_length": str(args.max_sequence_length),
        "index_0": "prompt",
        "index_1": "empty",
    }


def cache_is_valid(path: Path, expected: dict[str, str], max_length: int) -> bool:
    if not path.is_file():
        return False
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            if handle.metadata() != expected:
                return False
            keys = set(handle.keys())
            if keys != {"pooled_prompt_embeds", "prompt_embeds", "text_ids"}:
                return False
            prompt_shape = tuple(handle.get_tensor("prompt_embeds").shape)
            pooled_shape = tuple(handle.get_tensor("pooled_prompt_embeds").shape)
            text_ids_shape = tuple(handle.get_tensor("text_ids").shape)
        return (
            prompt_shape[0:2] == (2, max_length)
            and pooled_shape[0] == 2
            and text_ids_shape == (max_length, 3)
        )
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    expected_metadata = metadata(args)
    if not args.force and cache_is_valid(output, expected_metadata, args.max_sequence_length):
        print(f"OminiControl prompt cache already valid: {output}")
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    prompts = [args.prompt, ""]

    clip_tokenizer = CLIPTokenizer.from_pretrained(
        args.flux_path, subfolder="tokenizer", local_files_only=True
    )
    clip_encoder = CLIPTextModel.from_pretrained(
        args.flux_path,
        subfolder="text_encoder",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval().to(device)
    clip_inputs = clip_tokenizer(
        prompts,
        padding="max_length",
        max_length=clip_tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        pooled_prompt_embeds = clip_encoder(
            clip_inputs.input_ids.to(device), output_hidden_states=False
        ).pooler_output.to(dtype=dtype).cpu().contiguous()
    del clip_encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    del clip_tokenizer, clip_inputs

    t5_tokenizer = T5TokenizerFast.from_pretrained(
        args.flux_path, subfolder="tokenizer_2", local_files_only=True
    )
    t5_encoder = T5EncoderModel.from_pretrained(
        args.flux_path,
        subfolder="text_encoder_2",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval().to(device)
    t5_inputs = t5_tokenizer(
        prompts,
        padding="max_length",
        max_length=args.max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        prompt_embeds = t5_encoder(
            t5_inputs.input_ids.to(device), output_hidden_states=False
        )[0].to(dtype=dtype).cpu().contiguous()
    del t5_encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tensors = {
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "text_ids": torch.zeros(args.max_sequence_length, 3, dtype=dtype),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    save_file(tensors, temporary, metadata=expected_metadata)
    os.replace(temporary, output)
    print(f"OminiControl prompt cache written: {output}")


if __name__ == "__main__":
    main()
