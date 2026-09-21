"""Compare a LoRA adapter against its base model on the held-out split.

Policy (docs/../data/corpus/05_finetuning_policy.md): an adapter may only be
promoted if format compliance improves and faithfulness does not drop by more
than two points. This script produces exactly that evidence.

Run:
    python finetune/eval_lora.py --adapter runs/lora/qwen3b-rag-style \
        --base Qwen/Qwen2.5-3B-Instruct
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

REFUSAL = "I cannot answer this from the available documents."
_CITATION = re.compile(r"\[S\d+\]")


def load_examples(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def generate(model, tokenizer, messages: list[dict], max_new_tokens: int) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy: the comparison must be reproducible
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


def score(rows: list[dict], predictions: list[str]) -> dict:
    cited = refused_ok = total_words = 0
    n_refusal_cases = 0
    for row, prediction in zip(rows, predictions, strict=True):
        expects_refusal = row.get("type") == "refusal"
        if expects_refusal:
            n_refusal_cases += 1
            refused_ok += int(REFUSAL.lower() in prediction.lower())
        else:
            cited += int(bool(_CITATION.search(prediction)))
        total_words += len(prediction.split())
    answerable = len(rows) - n_refusal_cases
    return {
        "examples": len(rows),
        "citation_rate": round(cited / answerable, 3) if answerable else None,
        "refusal_rate": round(refused_ok / n_refusal_cases, 3) if n_refusal_cases else None,
        "mean_answer_words": round(total_words / len(rows), 1) if rows else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/finetune/eval.jsonl"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_examples(args.dataset)
    prompts = [row["messages"][:-1] for row in rows]  # drop the gold assistant turn

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results: dict[str, dict] = {}
    samples: dict[str, list[str]] = {}

    for label in ("base", "adapter"):
        model = AutoModelForCausalLM.from_pretrained(
            args.base, dtype=torch.bfloat16, device_map="auto"
        )
        if label == "adapter":
            model = PeftModel.from_pretrained(model, str(args.adapter))
        model.eval()

        started = time.perf_counter()
        predictions = [generate(model, tokenizer, m, args.max_new_tokens) for m in prompts]
        results[label] = {
            **score(rows, predictions),
            "seconds": round(time.perf_counter() - started, 1),
        }
        samples[label] = predictions[:3]

        del model
        torch.cuda.empty_cache()

    verdict = "promote"
    base_rate = results["base"].get("citation_rate") or 0.0
    adapter_rate = results["adapter"].get("citation_rate") or 0.0
    if adapter_rate < base_rate:
        verdict = "reject: format compliance regressed"

    report = {
        "base_model": args.base,
        "adapter": str(args.adapter),
        "dataset": str(args.dataset),
        "results": results,
        "verdict": verdict,
        "samples": samples,
        "note": (
            "Format compliance only. Run `voicerag eval --judge` against the RAG "
            "eval set before promoting; faithfulness must not drop by more than 2 points."
        ),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    out = args.out or args.adapter / "eval_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
