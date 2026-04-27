"""Push trained artifacts (merged weights / GGUF) to the Hugging Face Hub."""

from __future__ import annotations

from pathlib import Path

from .config import PipelineConfig
from .model import load_model_and_tokenizer
from .secrets import load_env, get_required


def _resolve_username(hf_token: str) -> str:
    from huggingface_hub import whoami

    info = whoami(token=hf_token)
    name = info.get("name")
    if not name:
        raise RuntimeError("Could not determine Hugging Face username from HF_TOKEN.")
    return name


def _load_for_push(cfg: PipelineConfig):
    """Reload base model + locally-saved LoRA adapters before push.

    When `cfg.train.lora_save_dir` contains a trained adapter (i.e. an
    `adapter_config.json`), we load it directly via Unsloth's
    `FastLanguageModel.from_pretrained`, which transparently resolves the
    base model from the adapter config and attaches the adapter weights in
    one shot. This avoids the `PeftModel.load_adapter()` signature mismatch
    (which requires a positional `adapter_name`) and the duplicate-`default`
    adapter that arises from calling `get_peft_model` again on top.
    """

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    adapter_dir = Path(cfg.train.lora_save_dir)
    has_adapter = (
        adapter_dir.is_dir() and (adapter_dir / "adapter_config.json").is_file()
    )

    if has_adapter:
        print(f"[push] Loading base model + LoRA adapters from {adapter_dir}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter_dir),
            max_seq_length=cfg.model.max_seq_length,
            load_in_4bit=cfg.model.load_in_4bit,
            load_in_8bit=cfg.model.load_in_8bit,
        )
        tokenizer = get_chat_template(tokenizer, chat_template=cfg.model.chat_template)
    else:
        print(
            f"[push] WARNING: No adapter_config.json found at {adapter_dir}; "
            "pushing the freshly-initialized LoRA model."
        )
        model, tokenizer = load_model_and_tokenizer(cfg.model)

    FastLanguageModel.for_inference(model)
    return model, tokenizer


def push_merged(cfg: PipelineConfig) -> str:
    load_env()
    hf_token = get_required(cfg.hf_token_env)
    username = _resolve_username(hf_token)

    model, tokenizer = _load_for_push(cfg)

    repo_id = f"{username}/{cfg.push.merged_repo_suffix}"
    print(f"[push] push_to_hub_merged -> {repo_id}")
    model.push_to_hub_merged(
        repo_id,
        tokenizer,
        save_method=cfg.push.merged_save_method,
        token=hf_token,
    )
    print(f"[push] Uploaded to https://huggingface.co/{repo_id}")
    return repo_id


def push_gguf(cfg: PipelineConfig) -> str:
    load_env()
    hf_token = get_required(cfg.hf_token_env)
    username = _resolve_username(hf_token)

    model, tokenizer = _load_for_push(cfg)

    repo_id = f"{username}/{cfg.push.gguf_repo_suffix}"
    print(f"[push] push_to_hub_gguf -> {repo_id}")
    model.push_to_hub_gguf(
        repo_id,
        tokenizer,
        quantization_method=cfg.push.gguf_quantization_methods,
        token=hf_token,
    )
    print(f"[push] Uploaded to https://huggingface.co/{repo_id}")
    return repo_id
