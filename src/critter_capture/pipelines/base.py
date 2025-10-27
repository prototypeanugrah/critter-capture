"""Base classes for pipelines."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from critter_capture.config import PipelineConfig, load_config
from critter_capture.utils import configure_logging, seed_everything

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    config: PipelineConfig
    workdir: Path
    run_metadata: Dict[str, Any]


class PipelineBase:
    """Base pipeline providing common lifecycle hooks."""

    def __init__(self, context: PipelineContext) -> None:
        self.context = context
        configure_logging(
            level=context.config.logging.level,
            log_dir=context.config.logging.log_dir,
            log_file=context.config.logging.log_file,
        )
        seed_everything(context.config.training.seed)
        LOGGER.info(
            "Pipeline initialized with environment %s", context.config.environment
        )

    def run(self) -> Dict[str, Any]:
        raise NotImplementedError


def build_context(
    config_path: Path, environment: Optional[str] = None
) -> PipelineContext:
    config = load_config(config_path, environment)
    context = PipelineContext(
        config=config, workdir=config_path.parent, run_metadata={}
    )
    return context


__all__ = ["PipelineBase", "PipelineContext", "build_context"]
