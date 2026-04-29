"""Default preset matching the original 27B Opus configuration."""

from __future__ import annotations

from qwen_finetune.config import PipelineConfig


def build_config() -> PipelineConfig:
    return PipelineConfig()
