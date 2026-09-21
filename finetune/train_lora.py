"""QLoRA supervised fine-tuning on a single consumer GPU.

Run:
    python finetune/train_lora.py --config finetune/configs/lora_qwen3b.yaml

Requires the GPU stack: ``pip install -r requirements-ml.txt``.

The script logs peak VRAM and wall-clock time into the run directory alongside
the adapter, because "did it fit and how long did it take" is the first thing
anyone asks when reproducing a fine-tune.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("base_model", "output_dir", "dataset"):
        if key not in cfg:
            raise KeyError(f"{path}: missing required key {key!r}")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("finetune/configs/lora_qwen3b.yaml"))
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate config and dataset, load nothing"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_path = Path(cfg["dataset"])
    eval_path = Path(cfg.get("eval_dataset", "")) if cfg.get("eval_dataset") else None
    output_dir = Path(cfg["output_dir"])

    if not train_path.exists():
        raise SystemExit(
            f"Training set not found: {train_path}. Run finetune/prepare_dataset.py first."
        )

    if args.dry_run:
        rows = sum(
            1 for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
        print(
            json.dumps(
                {"config": str(args.config), "base_model": cfg["base_model"], "train_rows": rows},
                indent=2,
            )
        )
        return

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from trl import SFTConfig, SFTTrainer

    from voicerag.monitoring.resources import ResourceSampler

    set_seed(int(cfg.get("seed", 42)))
    quant = cfg.get("quantization", {})
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]

    bnb_config = None
    if quant.get("load_in_4bit"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(torch, quant.get("bnb_4bit_compute_dtype", "bfloat16")),
            bnb_4bit_use_double_quant=quant.get("bnb_4bit_use_double_quant", True),
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        # Qwen/Llama chat models ship without a pad token; reusing EOS is the standard fix.
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb_config,
        dtype=torch.bfloat16 if train_cfg.get("bf16", True) else torch.float16,
        device_map="auto",
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=train_cfg.get("gradient_checkpointing", True)
        )

    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=list(lora_cfg["target_modules"]),
        bias="none",
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )
    model = get_peft_model(model, peft_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    data_files = {"train": str(train_path)}
    if eval_path and eval_path.exists():
        data_files["eval"] = str(eval_path)
    dataset = load_dataset("json", data_files=data_files)

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 3)),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 16)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        max_length=int(train_cfg.get("max_seq_length", 2048)),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        bf16=bool(train_cfg.get("bf16", True)),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        eval_strategy=train_cfg.get("eval_strategy", "epoch") if "eval" in data_files else "no",
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        save_total_limit=int(train_cfg.get("save_total_limit", 2)),
        optim=train_cfg.get("optim", "paged_adamw_8bit"),
        report_to=train_cfg.get("report_to", "none"),
        seed=int(cfg.get("seed", 42)),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("eval"),
        processing_class=tokenizer,
    )

    started = time.perf_counter()
    with ResourceSampler(interval_s=1.0) as sampler:
        result = trainer.train()
    elapsed = time.perf_counter() - started

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    adapter_mb = sum(f.stat().st_size for f in output_dir.rglob("*.safetensors")) / 1024**2
    run_card = {
        "base_model": cfg["base_model"],
        "config": str(args.config),
        "train_examples": len(dataset["train"]),
        "eval_examples": len(dataset["eval"]) if "eval" in data_files else 0,
        "train_runtime_s": round(elapsed, 1),
        "train_loss": round(float(result.training_loss), 4),
        "trainable_params": trainable,
        "trainable_percent": round(100 * trainable / total, 4),
        "adapter_size_mb": round(adapter_mb, 1),
        "peak_resources": sampler.summary(),
        "lora": lora_cfg,
        "training": train_cfg,
    }
    (output_dir / "run_card.json").write_text(json.dumps(run_card, indent=2), encoding="utf-8")
    print(json.dumps(run_card, indent=2))
    print(f"\nAdapter saved to {output_dir}. Evaluate it before promoting:")
    print(f"  python finetune/eval_lora.py --adapter {output_dir} --base {cfg['base_model']}")


if __name__ == "__main__":
    main()
