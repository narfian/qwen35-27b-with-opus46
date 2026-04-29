"""Copy this preset when creating a fully-customized configuration."""

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
            model_name="unsloth/Qwen3.5-27B",
            max_seq_length=32768,
            load_in_4bit=True,
            load_in_8bit=False,
            full_finetuning=False,
            lora_r=64,
            lora_alpha=64,
            lora_dropout=0.0,
            lora_bias="none",
            lora_target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "out_proj",
            ],
            use_gradient_checkpointing="unsloth",
            use_rslora=False,
            random_state=3407,
            chat_template="qwen3-thinking",
        ),
        data=DataConfig(
            random_seed=12181531,
            max_context_window=8192,
            num_samples={
                "ds1": 3900,
                "ds2": 700,
                "ds3": 9633,
            },
            ds1_name="nohurry/Opus-4.6-Reasoning-3000x-filtered",
            ds2_name="Jackrong/Qwen3.5-reasoning-700x",
            ds3_name="Roman1111111/claude-opus-4.6-10000x",
        ),
        train=TrainConfig(
            output_dir="./checkpoints/Qwen3.5-27B",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=6,
            warmup_ratio=0.05,
            num_train_epochs=2.0,
            max_steps=-1,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            save_steps=200,
            save_total_limit=1,
            save_strategy="steps",
            report_to="wandb",
            train_on_responses_only=True,
            lora_save_dir="./qwen35_27b_opus_lora",
        ),
        push=PushConfig(
            merged_repo_suffix="Qwen3.5-27B-opus",
            gguf_repo_suffix="Qwen3.5-27B-opus-GGUF",
            merged_save_method="merged_16bit",
            gguf_quantization_methods=[
                "q4_k_m",
                "q8_0",
                "bf16",
            ],
        ),
        hf_token_env="HF_TOKEN",
        wandb_api_key_env="WANDB_API_KEY",
        wandb_project="qwen35-27b-with-opus46",
    )
