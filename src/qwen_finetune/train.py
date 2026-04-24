"""Training entry point."""

from __future__ import annotations

import os
from pathlib import Path

from .config import PipelineConfig
from .data import build_dataset
from .model import load_model_and_tokenizer
from .secrets import get_optional, get_required, load_env


def _setup_wandb(cfg: PipelineConfig) -> None:
    if cfg.train.report_to != "wandb":
        return

    import wandb

    wandb_key = get_optional(cfg.wandb_api_key_env)
    if not wandb_key:
        raise RuntimeError(
            f"`report_to=wandb` but `{cfg.wandb_api_key_env}` is not set. "
            f"Either set it in `.env` or pass --report-to=none."
        )
    wandb.login(key=wandb_key)

    if cfg.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)


def run_training(cfg: PipelineConfig) -> None:
    load_env()
    _setup_wandb(cfg)

    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] Loading model: {cfg.model.model_name}")
    model, tokenizer = load_model_and_tokenizer(cfg.model)

    print("[train] Preparing dataset...")
    dataset = build_dataset(cfg.data, tokenizer)
    print(f"[train] Final dataset size: {len(dataset)}")

    from trl import SFTConfig, SFTTrainer

    sft_args = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        warmup_ratio=cfg.train.warmup_ratio,
        num_train_epochs=cfg.train.num_train_epochs,
        max_steps=cfg.train.max_steps,
        learning_rate=cfg.train.learning_rate,
        logging_steps=cfg.train.logging_steps,
        optim=cfg.train.optim,
        weight_decay=cfg.train.weight_decay,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        seed=cfg.train.seed,
        save_steps=cfg.train.save_steps,
        save_total_limit=cfg.train.save_total_limit,
        save_strategy=cfg.train.save_strategy,
        report_to=cfg.train.report_to,
        output_dir=str(output_dir),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=sft_args,
    )

    if cfg.train.train_on_responses_only:
        from unsloth.chat_templates import train_on_responses_only

        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n<think>",
        )

    print("[train] Starting trainer.train() ...")
    trainer.train()

    save_dir = Path(cfg.train.lora_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] Saving LoRA adapters to {save_dir}")
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    print("[train] Done.")


def ensure_hf_login(cfg: PipelineConfig) -> str:
    load_env()
    return get_required(
        cfg.hf_token_env, hint="needed to push artifacts to the HF Hub"
    )
