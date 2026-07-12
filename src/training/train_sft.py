from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import DatasetDict, load_dataset
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
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def validate_messages_example(example: dict[str, Any]) -> bool:
    messages = example.get("messages")

    if not isinstance(messages, list) or len(messages) < 2:
        return False

    for message in messages:
        if not isinstance(message, dict):
            return False

        if "role" not in message or "content" not in message:
            return False

        if not isinstance(message["role"], str):
            return False

        if not isinstance(message["content"], str):
            return False

        if not message["content"].strip():
            return False

    return True


def limit_dataset_samples(
    dataset: DatasetDict,
    max_train_samples: int | None,
    max_eval_samples: int | None,
) -> DatasetDict:
    if max_train_samples is not None and max_train_samples > 0:
        train_count = min(max_train_samples, len(dataset["train"]))
        dataset["train"] = dataset["train"].select(range(train_count))

    if max_eval_samples is not None and max_eval_samples > 0:
        eval_count = min(max_eval_samples, len(dataset["validation"]))
        dataset["validation"] = dataset["validation"].select(range(eval_count))

    return dataset


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    return str(value)


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        ),
        encoding="utf-8",
    )


def main() -> None:
    params = load_params()

    cfg = params["sft"]
    qcfg = cfg["qlora"]
    lcfg = cfg["lora"]

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    output_dir = Path(cfg["output_dir"])
    train_results_path = Path(cfg["train_results_path"])
    eval_results_path = Path(cfg["eval_results_path"])

    output_dir.mkdir(parents=True, exist_ok=True)
    train_results_path.parent.mkdir(parents=True, exist_ok=True)
    eval_results_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "json",
        data_files={
            "train": cfg["train_path"],
            "validation": cfg["val_path"],
        },
    )

    max_train_samples_value = cfg.get("max_train_samples")
    max_eval_samples_value = cfg.get("max_eval_samples")

    max_train_samples = (
        int(max_train_samples_value)
        if max_train_samples_value is not None
        else None
    )
    max_eval_samples = (
        int(max_eval_samples_value)
        if max_eval_samples_value is not None
        else None
    )

    dataset = limit_dataset_samples(
        dataset=dataset,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
    )

    bad_train = [
        index
        for index, example in enumerate(dataset["train"])
        if not validate_messages_example(example)
    ]
    bad_val = [
        index
        for index, example in enumerate(dataset["validation"])
        if not validate_messages_example(example)
    ]

    if bad_train:
        raise ValueError(
            f"Invalid train examples at indices: {bad_train[:10]}"
        )

    if bad_val:
        raise ValueError(
            f"Invalid validation examples at indices: {bad_val[:10]}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"],
        use_fast=True,
        trust_remote_code=bool(cfg.get("trust_remote_code", False)),
    )

    eos_token = cfg.get("eos_token")

    if eos_token:
        tokenizer.eos_token = eos_token

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype_name = qcfg.get(
        "bnb_4bit_compute_dtype",
        "bfloat16",
    )

    if not hasattr(torch, compute_dtype_name):
        raise ValueError(
            f"Unsupported torch dtype: {compute_dtype_name}"
        )

    compute_dtype = getattr(torch, compute_dtype_name)

    quant_config = None

    if bool(qcfg.get("enabled", False)):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=bool(qcfg.get("load_in_4bit", True)),
            bnb_4bit_quant_type=qcfg.get(
                "bnb_4bit_quant_type",
                "nf4",
            ),
            bnb_4bit_use_double_quant=bool(
                qcfg.get("bnb_4bit_use_double_quant", True)
            ),
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=quant_config,
        trust_remote_code=bool(
            cfg.get("trust_remote_code", False)
        ),
        dtype=compute_dtype,
        device_map=cfg.get("device_map", "auto"),
    )

    model.config.use_cache = False

    if eos_token and tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id

        if getattr(model, "generation_config", None) is not None:
            model.generation_config.eos_token_id = (
                tokenizer.eos_token_id
            )

    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

        if getattr(model, "generation_config", None) is not None:
            model.generation_config.pad_token_id = (
                tokenizer.pad_token_id
            )

    peft_config = LoraConfig(
        r=int(lcfg.get("r", 16)),
        lora_alpha=int(lcfg.get("lora_alpha", 32)),
        lora_dropout=float(lcfg.get("lora_dropout", 0.05)),
        bias=lcfg.get("bias", "none"),
        task_type=lcfg.get("task_type", "CAUSAL_LM"),
        target_modules=lcfg.get(
            "target_modules",
            "all-linear",
        ),
        use_rslora=bool(lcfg.get("use_rslora", False)),
    )

    eval_strategy = str(
        cfg.get("eval_strategy", "steps")
    )
    save_strategy = str(
        cfg.get("save_strategy", "steps")
    )
    logging_strategy = str(
        cfg.get("logging_strategy", "steps")
    )

    run_final_eval = bool(
        cfg.get(
            "run_final_eval",
            eval_strategy != "no",
        )
    )

    load_best_model_at_end = bool(
        cfg.get("load_best_model_at_end", True)
    )

    if eval_strategy == "no" or save_strategy == "no":
        load_best_model_at_end = False

    sft_config = SFTConfig(
        output_dir=str(output_dir),

        max_length=int(
            cfg.get("max_seq_length", 1024)
        ),

        max_steps=int(cfg.get("max_steps", -1)),
        num_train_epochs=float(
            cfg.get("num_train_epochs", 5)
        ),

        learning_rate=float(
            cfg.get("learning_rate", 2e-4)
        ),

        per_device_train_batch_size=int(
            cfg.get("per_device_train_batch_size", 2)
        ),
        per_device_eval_batch_size=int(
            cfg.get("per_device_eval_batch_size", 2)
        ),
        gradient_accumulation_steps=int(
            cfg.get("gradient_accumulation_steps", 8)
        ),

        logging_strategy=logging_strategy,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,

        logging_steps=int(cfg.get("logging_steps", 10)),
        eval_steps=int(cfg.get("eval_steps", 100)),
        save_steps=int(cfg.get("save_steps", 100)),
        save_total_limit=int(
            cfg.get("save_total_limit", 2)
        ),

        warmup_ratio=float(
            cfg.get("warmup_ratio", 0.03)
        ),
        lr_scheduler_type=cfg.get(
            "lr_scheduler_type",
            "cosine",
        ),
        weight_decay=float(
            cfg.get("weight_decay", 0.01)
        ),
        max_grad_norm=float(
            cfg.get("max_grad_norm", 1.0)
        ),

        bf16=bool(cfg.get("bf16", True)),
        fp16=bool(cfg.get("fp16", False)),
        tf32=bool(cfg.get("tf32", True)),

        gradient_checkpointing=bool(
            cfg.get("gradient_checkpointing", True)
        ),

        assistant_only_loss=bool(
            cfg.get("assistant_only_loss", True)
        ),

        packing=bool(cfg.get("packing", False)),
        eval_packing=bool(
            cfg.get("eval_packing", False)
        ),

        dataset_num_proc=int(
            cfg.get("dataset_num_proc", 1)
        ),
        dataloader_num_workers=int(
            cfg.get("dataloader_num_workers", 0)
        ),

        report_to=cfg.get("report_to", "none"),

        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=cfg.get(
            "metric_for_best_model",
            "eval_loss",
        ),
        greater_is_better=bool(
            cfg.get("greater_is_better", False)
        ),

        eos_token=eos_token,
        seed=seed,
    )

    callbacks = []

    early_stopping_patience = int(
        cfg.get("early_stopping_patience", 0)
    )

    if (
        eval_strategy != "no"
        and early_stopping_patience > 0
    ):
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=float(
                    cfg.get(
                        "early_stopping_threshold",
                        0.0,
                    )
                ),
            )
        )

    trainer_eval_dataset = (
        dataset["validation"]
        if eval_strategy != "no" or run_final_eval
        else None
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=trainer_eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
    )

    train_result = trainer.train()

    save_model_after_train = bool(
        cfg.get("save_model_after_train", True)
    )

    if save_model_after_train:
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

    train_metrics = dict(train_result.metrics)
    train_metrics["train_samples"] = int(
        len(dataset["train"])
    )
    train_metrics["configured_max_steps"] = int(
        cfg.get("max_steps", -1)
    )
    train_metrics["per_device_train_batch_size"] = int(
        cfg.get("per_device_train_batch_size", 1)
    )
    train_metrics["gradient_accumulation_steps"] = int(
        cfg.get("gradient_accumulation_steps", 1)
    )

    if trainer.is_world_process_zero():
        trainer.log_metrics("train", train_metrics)
        save_json(train_results_path, train_metrics)

    if run_final_eval:
        eval_metrics = trainer.evaluate()
        eval_metrics["eval_samples"] = int(
            len(dataset["validation"])
        )
    else:
        eval_metrics = {
            "eval_skipped": 1,
            "eval_samples": 0,
        }

    if trainer.is_world_process_zero():
        if run_final_eval:
            trainer.log_metrics("eval", eval_metrics)

        save_json(eval_results_path, eval_metrics)

        print(
            json.dumps(
                {
                    "status": "ok",
                    "output_dir": str(output_dir),
                    "train_results_path": str(
                        train_results_path
                    ),
                    "eval_results_path": str(
                        eval_results_path
                    ),
                    "train_samples": len(dataset["train"]),
                    "eval_samples": (
                        len(dataset["validation"])
                        if run_final_eval
                        else 0
                    ),
                    "max_steps": int(
                        cfg.get("max_steps", -1)
                    ),
                    "final_eval_run": run_final_eval,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()