"""Secret / credential loading helpers.

Credentials (HF_TOKEN, WANDB_API_KEY, ...) are loaded from a local `.env`
file so they are never committed to git. See `.env.example` for the format.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def load_env(env_file: Optional[str] = None) -> None:
    """Load environment variables from the project `.env` file.

    The search order is:
        1. Explicit path passed in via `env_file`.
        2. `./.env` in the current working directory.
        3. `<repo_root>/.env` (inferred from this file's location).
    Already-set variables in the real environment take precedence.
    """

    candidates = []
    if env_file:
        candidates.append(Path(env_file))

    candidates.append(Path.cwd() / ".env")

    repo_root = Path(__file__).resolve().parents[2]
    candidates.append(repo_root / ".env")

    for candidate in candidates:
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            return


def get_required(name: str, hint: Optional[str] = None) -> str:
    value = os.environ.get(name)
    if not value:
        hint_msg = f" ({hint})" if hint else ""
        raise RuntimeError(
            f"Environment variable `{name}` is not set{hint_msg}. "
            f"Add it to your `.env` file (see `.env.example`) or export it "
            f"in your shell before running."
        )
    return value


def get_optional(name: str) -> Optional[str]:
    value = os.environ.get(name)
    return value or None
