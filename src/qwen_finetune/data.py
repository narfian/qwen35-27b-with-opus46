"""Dataset preparation for Qwen 3.5 SFT training.

Implements the same sample / normalize / template / filter pipeline as the
reference Colab notebook, but as an importable function that can be reused
from a CLI entry point.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import re
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset

from .config import DataConfig

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def _strip(x: Optional[str]) -> str:
    return (x or "").strip()


def normalize_assistant_to_think_solution(text: str) -> str:
    text = _strip(text)
    if not text:
        return "<think></think>\n"

    m = THINK_BLOCK_RE.search(text)
    if m:
        think_block = m.group(0).strip()
        rest = text[m.end() :].lstrip()
        return f"{think_block}\n{rest}".rstrip() if rest else f"{think_block}\n"

    return f"<think></think>\n{text}".rstrip()


def build_assistant_with_reasoning(content: str, reasoning: str = "") -> str:
    content = _strip(content)
    reasoning = _strip(reasoning)

    if "<think>" in content and "</think>" in content:
        return normalize_assistant_to_think_solution(content)

    if reasoning:
        if content:
            return f"<think>{reasoning}</think>\n{content}"
        return f"<think>{reasoning}</think>\n"

    return normalize_assistant_to_think_solution(content)


def parse_message_item(m: Any) -> Optional[Dict[str, Any]]:
    if isinstance(m, dict):
        return m
    if isinstance(m, str):
        s = m.strip()
        if not s:
            return None
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def _load_ds3_via_pandas_parquet(dataset_name: str) -> Dataset:
    parquet_path = (
        f"hf://datasets/{dataset_name}"
        "@refs/convert/parquet/default/train/0000.parquet"
    )
    df = pd.read_parquet(parquet_path)
    return Dataset.from_pandas(df, preserve_index=False)


def load_and_sample(
    dataset_name: str,
    sample_count: Optional[int] = None,
    split: str = "train",
    subset: Optional[str] = None,
    random_seed: int = 12181531,
) -> Dataset:
    try:
        if subset:
            ds = load_dataset(dataset_name, subset, split=split)
        else:
            ds = load_dataset(dataset_name, split=split)
    except ValueError as e:
        err = str(e)
        if (
            dataset_name == "Roman1111111/claude-opus-4.6-10000x"
            and "Feature type 'Json' not found" in err
        ):
            ds = _load_ds3_via_pandas_parquet(dataset_name)
        else:
            raise

    if sample_count is not None:
        sample_count = min(sample_count, len(ds))
        ds = ds.shuffle(seed=random_seed).select(range(sample_count))

    return ds


def _format_ds1(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    problems = examples.get("problem", [])
    thinkings = examples.get("thinking", [])
    solutions = examples.get("solution", [])

    out: List[List[Dict[str, str]]] = []
    for p, t, s in zip(problems, thinkings, solutions):
        p = _strip(p)
        t = _strip(t)
        s = _strip(s)
        if not p or not s:
            continue

        assistant = f"<think>{t}</think>\n{s}" if t else f"<think></think>\n{s}"
        out.append(
            [
                {"role": "user", "content": p},
                {"role": "assistant", "content": assistant},
            ]
        )

    return {"conversations": out}


def _format_ds2(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    convos_list = examples.get("conversation", [])
    out: List[List[Dict[str, str]]] = []

    for conv in convos_list:
        if not conv:
            continue

        cleaned: List[Dict[str, str]] = []
        for m in conv:
            frm = (m.get("from") or "").strip()
            val = m.get("value", "")
            if frm == "human":
                cleaned.append({"role": "user", "content": _strip(val)})
            elif frm == "gpt":
                cleaned.append(
                    {
                        "role": "assistant",
                        "content": normalize_assistant_to_think_solution(val),
                    }
                )

        if len(cleaned) < 2 or cleaned[-1]["role"] != "assistant":
            continue
        out.append(cleaned)

    return {"conversations": out}


def _format_ds3(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    messages_list = examples.get("messages", [])
    out: List[List[Dict[str, str]]] = []

    for msgs in messages_list:
        if not msgs:
            continue

        parsed_msgs: List[Dict[str, Any]] = []
        for m in msgs:
            pm = parse_message_item(m)
            if pm is not None:
                parsed_msgs.append(pm)
        if not parsed_msgs:
            continue

        convo = [m for m in parsed_msgs if m.get("role") != "system"]
        if len(convo) < 2 or convo[-1].get("role") != "assistant":
            continue

        cleaned: List[Dict[str, str]] = []
        for m in convo:
            role = m.get("role")
            content = m.get("content", "")
            reasoning = m.get("reasoning", "")

            if role == "assistant":
                content = build_assistant_with_reasoning(content, reasoning)
            else:
                content = _strip(content)

            if role in ("user", "assistant") and content is not None:
                cleaned.append({"role": role, "content": content})

        if len(cleaned) < 2 or cleaned[-1]["role"] != "assistant":
            continue
        out.append(cleaned)

    return {"conversations": out}


def build_dataset(cfg: DataConfig, tokenizer) -> Dataset:
    """Build the shuffled, filtered SFT dataset with a `text` column."""

    ds1 = load_and_sample(
        cfg.ds1_name, cfg.num_samples["ds1"], random_seed=cfg.random_seed
    )
    ds2 = load_and_sample(
        cfg.ds2_name, cfg.num_samples["ds2"], random_seed=cfg.random_seed
    )
    ds3 = load_and_sample(
        cfg.ds3_name, cfg.num_samples["ds3"], random_seed=cfg.random_seed
    )

    ds1 = ds1.map(_format_ds1, batched=True, remove_columns=ds1.column_names)
    ds2 = ds2.map(_format_ds2, batched=True, remove_columns=ds2.column_names)
    ds3 = ds3.map(_format_ds3, batched=True, remove_columns=ds3.column_names)

    ds1 = ds1.filter(lambda x: x["conversations"] is not None and len(x["conversations"]) > 0)
    ds2 = ds2.filter(lambda x: x["conversations"] is not None and len(x["conversations"]) > 0)
    ds3 = ds3.filter(lambda x: x["conversations"] is not None and len(x["conversations"]) > 0)

    combined = concatenate_datasets([ds1, ds2, ds3]).shuffle(seed=cfg.random_seed)

    def _apply_chat_template(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]
        return {"text": texts}

    dataset = combined.map(_apply_chat_template, batched=True)

    num_proc = mp.cpu_count()
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    max_len = cfg.max_context_window

    def _filter_long(examples):
        texts = examples["text"]
        tokenized = text_tok(
            texts, truncation=False, padding=False, add_special_tokens=False
        )["input_ids"]
        return [len(toks) <= max_len for toks in tokenized]

    dataset = dataset.filter(_filter_long, batched=True, num_proc=num_proc)

    dataset = dataset.filter(
        lambda x: all(
            (m["role"] != "assistant")
            or (("<think>" in m["content"]) and ("</think>\n" in m["content"]))
            for m in x["conversations"]
        )
    )

    return dataset
