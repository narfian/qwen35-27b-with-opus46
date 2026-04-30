"""Configuration objects and preset loading for the fine-tuning pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from pkgutil import iter_modules
from typing import Dict, List, Optional


CONFIG_PRESET_PACKAGE = f"{__package__}.config_presets"


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
    per_device_train_batch_size: int = 1
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
    lora_save_dir: str = "./qwen35_27b_opus_lora"

    # Runtime patch toggles. Default True keeps Ampere (A100) compatibility for
    # FLA's chunk_gated_delta_rule kernels (otherwise Triton raises
    # OutOfResources: shared memory at num_stages>=3). Set to False to keep
    # FLA's stock autotune candidates (recommended only on Hopper/Blackwell).
    fla_ampere_safe_autotune: bool = True


@dataclass
class PushConfig:
    merged_repo_suffix: str = "Qwen3.5-27B-opus"
    gguf_repo_suffix: str = "Qwen3.5-27B-opus-GGUF"
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


def _normalize_preset_name(name: str) -> str:
    return name.strip().replace("-", "_")


def list_config_presets() -> list[str]:
    """Return available config preset names for CLI choices/help."""

    package = import_module(CONFIG_PRESET_PACKAGE)
    return sorted(
        module.name.replace("_", "-")
        for module in iter_modules(package.__path__)
        if not module.name.startswith("_")
    )


def load_config_preset(name: str = "default") -> PipelineConfig:
    module_name = _normalize_preset_name(name)

    qualified_module_name = f"{CONFIG_PRESET_PACKAGE}.{module_name}"

    try:
        module = import_module(qualified_module_name)
    except ModuleNotFoundError as exc:
        if exc.name != qualified_module_name:
            raise
        available = ", ".join(list_config_presets())
        raise ValueError(
            f"Unknown config preset: {name!r}. Available presets: {available}"
        ) from exc

    try:
        cfg = module.build_config()
    except AttributeError as exc:
        raise ValueError(
            f"Config preset {name!r} must define build_config()."
        ) from exc

    if not isinstance(cfg, PipelineConfig):
        raise TypeError(
            f"Config preset {name!r} returned {type(cfg).__name__}, "
            "expected PipelineConfig."
        )

    return cfg


def default_config(preset: str = "default") -> PipelineConfig:
    return load_config_preset(preset)
