"""Configuration objects for the fine-tuning pipeline.

All defaults reproduce the values used in the reference Colab notebook
(`Qwopus3-5-27b-Colab.ipynb`). Override via CLI flags or by editing this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    model_name: str = "unsloth/Qwen3.5-27B"
    max_seq_length: int = 32768
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    full_finetuning: bool = False

    # LoRA / PEFT
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_bias: str = "none"
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "out_proj",
        ]
    )
    use_gradient_checkpointing: str = "unsloth"
    use_rslora: bool = False
    random_state: int = 3407

    # Chat template used after loading.
    chat_template: str = "qwen3-thinking"


@dataclass
class DataConfig:
    random_seed: int = 12181531
    max_context_window: int = 8192

    num_samples: Dict[str, int] = field(
        default_factory=lambda: {
            "ds1": 3900,  # nohurry/Opus-4.6-Reasoning-3000x-filtered
            "ds2": 700,  # Jackrong/Qwen3.5-reasoning-700x
            "ds3": 9633,  # Roman1111111/claude-opus-4.6-10000x
        }
    )

    ds1_name: str = "nohurry/Opus-4.6-Reasoning-3000x-filtered"
    ds2_name: str = "Jackrong/Qwen3.5-reasoning-700x"
    ds3_name: str = "Roman1111111/claude-opus-4.6-10000x"


@dataclass
class TrainConfig:
    output_dir: str = "./checkpoints/Qwen3.5-27B"
    per_device_train_batch_size: int = 6
    gradient_accumulation_steps: int = 6
    warmup_ratio: float = 0.05
    num_train_epochs: float = 2.0
    max_steps: int = -1
    learning_rate: float = 2e-4
    logging_steps: int = 1
    optim: str = "adamw_8bit"
    weight_decay: float = 0.001
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    save_steps: int = 200
    save_total_limit: int = 1
    save_strategy: str = "steps"
    report_to: str = "wandb"  # set to "none" to disable W&B
    train_on_responses_only: bool = True
    lora_save_dir: str = "./qwen_lora"


@dataclass
class PushConfig:
    merged_repo_suffix: str = "Qwopus3.5-27B"
    gguf_repo_suffix: str = "Qwopus3.5-27B-GGUF"
    merged_save_method: str = "merged_16bit"
    gguf_quantization_methods: List[str] = field(
        default_factory=lambda: ["q4_k_m", "q8_0", "bf16"]
    )


@dataclass
class PipelineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    push: PushConfig = field(default_factory=PushConfig)

    hf_token_env: str = "HF_TOKEN"
    wandb_api_key_env: str = "WANDB_API_KEY"
    wandb_project: Optional[str] = "qwen35-27b-with-opus46"


def default_config() -> PipelineConfig:
    return PipelineConfig()
