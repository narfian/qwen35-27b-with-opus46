"""Preset mirroring the Qwopus-3.5-35B-A3B Kaggle notebook setup.

Reproduces the exact training recipe used in
``Qwopus-3.5-35B-A3B-Kaggle.ipynb`` (Unsloth + LoRA SFT on the Opus reasoning
dataset mix), so that ``uv run qwen-finetune train --config-preset
q35-35b-a3b-kaggle`` produces an equivalent run outside Kaggle.
"""

from __future__ import annotations

from qwen_finetune.config import (
    DataConfig,
    ModelConfig,
    PipelineConfig,
    PushConfig,
    TrainConfig,
)


def build_config() -> PipelineConfig:
    return PipelineConfig(
        model=ModelConfig(
            model_name="unsloth/Qwen3.5-35B-A3B",
            max_seq_length=8192,
            load_in_4bit=True,
            load_in_8bit=False,
            full_finetuning=False,
            lora_r=32,
            lora_alpha=64,
            lora_dropout=0.0,
            lora_bias="none",
            lora_target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                ## LoRA 적용 대상에서 제외
                # "gate_proj",
                # "up_proj",
                # "down_proj",
                # "gate_up_proj",
            ],
            use_gradient_checkpointing="unsloth", #True,
            use_rslora=False,
            random_state=3407,
            chat_template="qwen3-thinking",
        ),
        data=DataConfig(
            random_seed=1234,
            max_context_window=8192,
            num_samples={
                "ds1": 3900,
                "ds2": 700,
                "ds3": 10000,
            },
            ds1_name="nohurry/Opus-4.6-Reasoning-3000x-filtered",
            ds2_name="Jackrong/Qwen3.5-reasoning-700x",
            ds3_name="Roman1111111/claude-opus-4.6-10000x",
        ),
        train=TrainConfig(
            output_dir="./checkpoints/Qwen-3.5-35B-A3B",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_ratio=0.04,
            num_train_epochs=2.0,
            max_steps=-1,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            save_steps=100,
            save_total_limit=1,
            save_strategy="steps",
            report_to="wandb",
            train_on_responses_only=True,
            lora_save_dir="./qwen_35b_a3b_opus_lora",
        ),
        push=PushConfig(
            merged_repo_suffix="Qwen3.5-35B-A3B-opus",
            gguf_repo_suffix="Qwen3.5-35B-A3B-opus-GGUF",
            merged_save_method="merged_16bit",
            gguf_quantization_methods=["q8_0"],
        ),
        hf_token_env="HF_TOKEN",
        wandb_api_key_env="WANDB_API_KEY",
        wandb_project="qwen3.5-35b-a3b-opus",
    )
