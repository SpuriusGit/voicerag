"""Merge a LoRA adapter into its base model and export standalone weights.

Needed when the adapter must be served by a runtime that cannot load PEFT
adapters at request time (vLLM without multi-LoRA, or Ollama via GGUF export).

Run:
    python finetune/merge_lora.py --adapter runs/lora/qwen3b-rag-style \
        --base Qwen/Qwen2.5-3B-Instruct --out runs/merged/qwen3b-rag-style
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Merging must happen in a float dtype: merging into 4-bit weights silently
    # degrades quality, so the base is reloaded at bf16 even if training was 4-bit.
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, str(args.adapter))
    merged = model.merge_and_unload()

    args.out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.out), safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(str(args.out))

    size_gb = sum(f.stat().st_size for f in args.out.rglob("*.safetensors")) / 1024**3
    print(f"Merged model written to {args.out} ({size_gb:.2f} GB)")
    print("To serve with Ollama, convert to GGUF with llama.cpp:")
    print(f"  python convert_hf_to_gguf.py {args.out} --outtype f16 --outfile model-f16.gguf")
    print("  ./llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M")


if __name__ == "__main__":
    main()
