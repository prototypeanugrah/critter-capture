"""Environment variable helpers."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def load_env_file(path: str | Path = ".env") -> None:
    """Load environment variables from a .env file if present."""

    env_path = Path(path)
    if env_path.exists():
        load_dotenv(env_path)


def require_env(keys: list[str]) -> dict[str, str]:
    """Ensure required environment variables are present and return them."""

    values = {}
    for key in keys:
        value = os.getenv(key)
        if not value:
            raise RuntimeError(f"Environment variable {key} must be set.")
        values[key] = value
    return values


__all__ = ["load_env_file", "require_env"]

