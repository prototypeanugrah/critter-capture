"""General IO helpers for file system operations."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text_file(path: Path, content: str) -> None:
    """Write text content to a file."""

    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def list_files(directory: Path, suffixes: Iterable[str] | None = None) -> list[Path]:
    """List files optionally filtered by suffixes."""

    if suffixes:
        return [path for path in directory.iterdir() if path.suffix in suffixes]
    return list(directory.iterdir())


__all__ = ["ensure_dir", "write_text_file", "list_files"]

