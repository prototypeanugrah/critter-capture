"""
Logging configuration helpers.

Provides a central entry-point to configure consistent structured logging across
pipelines and modules.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def configure_logging(
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure logging for the application.

    Parameters
    ----------
    level:
        Log level name (e.g., ``INFO``).
    log_dir:
        Directory where the log file should be created. The directory is created
        if it does not already exist.
    log_file:
        Log file name. When provided, a rotating file handler is attached in
        addition to the console handler.
    """

    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if log_dir and log_file:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=Path(log_dir) / log_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logging.getLogger().addHandler(file_handler)


__all__ = ["configure_logging"]

