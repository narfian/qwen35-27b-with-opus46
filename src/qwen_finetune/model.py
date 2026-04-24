"""Model + tokenizer loading utilities."""

from __future__ import annotations

from typing import Tuple

from .config import ModelConfig


def load_model_and_tokenizer(cfg: ModelConfig) -> Tuple[object, object]:
    """Load the base Qwen model + tokenizer and attach LoRA adapters."""

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        load_in_8bit=cfg.load_in_8bit,
        full_finetuning=cfg.full_finetuning,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        target_modules=cfg.lora_target_modules,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.lora_bias,
        use_gradient_checkpointing=cfg.use_gradient_checkpointing,
        random_state=cfg.random_state,
        use_rslora=cfg.use_rslora,
        loftq_config=None,
    )

    tokenizer = get_chat_template(tokenizer, chat_template=cfg.chat_template)
    return model, tokenizer
