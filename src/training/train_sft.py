from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    set_seed,
)
from trl import SFTConfig, SFTTrainer


def load_params(path: str = "params.yaml") -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_messages_example(example: dict[str, Any]) -> bool:
    messages = example.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    for msg in messages:
        if not isinstance(msg, dict):
            return False
        if "role" not in msg or "content" not in msg:
            return False
        if not isinstance(msg["role"], str) or not isinstance(msg["content"], str):
            return False
        if not msg["content"].strip():
            return False
    return True


def main() -> None:
    params = load_params()
    cfg = params["sft"]
    qcfg = cfg["qlora"]
    lcfg = cfg["lora"]

    set_seed(int(cfg.get("seed", 42)))
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "json",
        data_files={
            "train": cfg["train_path"],
            "validation": cfg["val_path"],
        },
    )

    bad_train = [i for i, ex in enumerate(dataset["train"]) if not validate_messages_example(ex)]
    bad_val = [i for i, ex in enumerate(dataset["validation"]) if not validate_messages_example(ex)]
    if bad_train:
        raise ValueError(f"Invalid train examples at indices: {bad_train[:10]}")
    if bad_val:
        raise ValueError(f"Invalid val examples at indices: {bad_val[:10]}")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"],
        use_fast=True,
        trust_remote_code=cfg.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = getattr(torch, qcfg.get("bnb_4bit_compute_dtype", "bfloat16"))

    quant_config = None
    if qcfg.get("enabled", False):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=bool(qcfg.get("load_in_4bit", True)),
            bnb_4bit_quant_type=qcfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=bool(qcfg.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=quant_config,
        trust_remote_code=cfg.get("trust_remote_code", False),
        torch_dtype=compute_dtype,
        device_map="auto",
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=int(lcfg.get("r", 16)),
        lora_alpha=int(lcfg.get("lora_alpha", 32)),
        lora_dropout=float(lcfg.get("lora_dropout", 0.05)),
        bias=lcfg.get("bias", "none"),
        task_type=lcfg.get("task_type", "CAUSAL_LM"),
        target_modules=lcfg.get("target_modules", "all-linear"),
    )

    sft_config = SFTConfig(
        output_dir=cfg["output_dir"],
        max_length=int(cfg.get("max_seq_length", 1024)),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        num_train_epochs=float(cfg.get("num_train_epochs", 5)),
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 2)),
        per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size", 2)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
        logging_steps=int(cfg.get("logging_steps", 10)),
        eval_steps=int(cfg.get("eval_steps", 100)),
        save_steps=int(cfg.get("save_steps", 100)),
        save_total_limit=int(cfg.get("save_total_limit", 2)),
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        bf16=bool(cfg.get("bf16", True)),
        fp16=bool(cfg.get("fp16", False)),
        tf32=bool(cfg.get("tf32", True)),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        assistant_only_loss=bool(cfg.get("assistant_only_loss", True)),
        packing=bool(cfg.get("packing", False)),
        eval_packing=bool(cfg.get("eval_packing", False)),
        dataset_num_proc=int(cfg.get("dataset_num_proc", 2)),
        report_to=cfg.get("report_to", "none"),
        load_best_model_at_end=bool(cfg.get("load_best_model_at_end", True)),
        metric_for_best_model=cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=bool(cfg.get("greater_is_better", False)),
        seed=int(cfg.get("seed", 42)),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=int(cfg.get("early_stopping_patience", 3)),
                early_stopping_threshold=float(cfg.get("early_stopping_threshold", 0.0)),
            )
        ],
    )

    train_result = trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    print(json.dumps(
        {
            "status": "ok",
            "output_dir": cfg["output_dir"],
            "train_samples": len(dataset["train"]),
            "eval_samples": len(dataset["validation"]),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()