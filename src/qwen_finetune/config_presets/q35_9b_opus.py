"""Qwen 3.5 9B preset for the Opus reasoning dataset mix."""

from __future__ import annotations

from qwen_finetune.config import ModelConfig, PipelineConfig, PushConfig, TrainConfig


def build_config() -> PipelineConfig:
    return PipelineConfig(
        model=ModelConfig(
            model_name="unsloth/Qwen3.5-9B",
        ),
        train=TrainConfig(
            output_dir="./checkpoints/Qwen3.5-9B",
            per_device_train_batch_size=3,
            gradient_accumulation_steps=12,
            lora_save_dir="./qwen35_9b_opus_lora",
        ),
        push=PushConfig(
            merged_repo_suffix="Qwen3.5-9B-opus",
            gguf_repo_suffix="Qwen3.5-9B-opus-GGUF",
        ),
    )
