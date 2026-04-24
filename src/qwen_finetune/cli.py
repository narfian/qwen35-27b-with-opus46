"""Command-line entry point.

Usage (from the repo root):

    uv run qwen-finetune train
    uv run qwen-finetune push-merged
    uv run qwen-finetune push-gguf
    uv run qwen-finetune prepare-data        # dry-run data pipeline only
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields, is_dataclass
from typing import Any

from .config import PipelineConfig, default_config


def _add_dataclass_args(parser: argparse.ArgumentParser, dc: Any, prefix: str) -> None:
    """Flatten a dataclass into `--group-field` CLI arguments.

    Only scalar fields (no lists / dicts) are exposed; complex overrides should
    be done by editing `config.py` directly.
    """

    for f in fields(dc):
        if f.type in ("List[str]", "Dict[str, int]", "List[int]"):
            continue
        if isinstance(getattr(dc, f.name), (list, dict)):
            continue

        cli_name = f"--{prefix}-{f.name.replace('_', '-')}"
        default = getattr(dc, f.name)

        kwargs: dict = {"default": default, "help": f"(default: {default})"}
        if isinstance(default, bool):
            kwargs["type"] = lambda v: str(v).lower() in ("1", "true", "yes", "y")
        elif default is None:
            kwargs["type"] = str
        else:
            kwargs["type"] = type(default)

        parser.add_argument(cli_name, dest=f"{prefix}__{f.name}", **kwargs)


def _apply_overrides(cfg: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    for key, value in vars(args).items():
        if "__" not in key:
            continue
        group, field_name = key.split("__", 1)
        sub = getattr(cfg, group, None)
        if sub is None or not is_dataclass(sub):
            continue
        if hasattr(sub, field_name):
            setattr(sub, field_name, value)
    return cfg


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-finetune",
        description="Qwen 3.5 27B SFT fine-tuning pipeline (Unsloth + LoRA).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cfg = default_config()

    for name, help_text in (
        ("train", "Run the full SFT training pipeline."),
        ("push-merged", "Push merged 16-bit weights to the HF Hub."),
        ("push-gguf", "Export and push GGUF quantizations to the HF Hub."),
        ("prepare-data", "Load, format, and filter the dataset (no training)."),
    ):
        p = sub.add_parser(name, help=help_text)
        _add_dataclass_args(p, cfg.model, "model")
        _add_dataclass_args(p, cfg.data, "data")
        _add_dataclass_args(p, cfg.train, "train")
        _add_dataclass_args(p, cfg.push, "push")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg = default_config()
    cfg = _apply_overrides(cfg, args)

    if args.command == "train":
        from .train import run_training

        run_training(cfg)
    elif args.command == "push-merged":
        from .push import push_merged

        push_merged(cfg)
    elif args.command == "push-gguf":
        from .push import push_gguf

        push_gguf(cfg)
    elif args.command == "prepare-data":
        from .data import build_dataset
        from .model import load_model_and_tokenizer

        _, tokenizer = load_model_and_tokenizer(cfg.model)
        ds = build_dataset(cfg.data, tokenizer)
        print(f"Final dataset size: {len(ds)}")
        if len(ds) > 0:
            print("=== Sample text (truncated) ===")
            print(ds[0]["text"][:800])
    else:  # pragma: no cover
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
