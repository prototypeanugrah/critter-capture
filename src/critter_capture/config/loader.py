"""
Configuration loading utilities.

Provides helpers to parse YAML configuration files into strongly typed Pydantic
objects. Supports environment-specific overrides and environment variable
interpolation for sensitive values.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml
from yaml.loader import SafeLoader

from .schema import PipelineConfig


def _deep_merge(
    base: Dict[str, Any],
    override: Dict[str, Any],
) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base``."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            base[key] = _deep_merge(dict(base[key]), value)
        else:
            base[key] = value
    return base


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=SafeLoader) or {}


def _expand_and_resolve_paths(
    data: Any,
    base_dir: Path,
    key_path: Tuple[str, ...] = (),
) -> Any:
    if isinstance(data, dict):
        return {
            key: _expand_and_resolve_paths(value, base_dir, key_path + (key,))
            for key, value in data.items()
        }
    if isinstance(data, list):
        return [_expand_and_resolve_paths(item, base_dir, key_path) for item in data]
    if isinstance(data, str):
        expanded = os.path.expandvars(data)
        last_key = key_path[-1] if key_path else ""
        if last_key.endswith("_path") or last_key.endswith("_dir"):
            path_obj = Path(expanded)
            if not path_obj.is_absolute():
                return str((base_dir / path_obj).resolve())
            return str(path_obj)
        return expanded
    return data


def load_config(
    path: Path | str,
    environment: Optional[str] = None,
) -> PipelineConfig:
    """
    Load the pipeline configuration.

    Parameters
    ----------
    path:
        Base path to the YAML configuration file.
    environment:
        Optional environment name (e.g., ``local``, ``ci``, ``prod``). When
        provided, an environment-specific section ``environments.<name>`` is
        merged on top of the base configuration.
    """

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    raw_config = _load_yaml(config_path)
    env_name = (
        environment or os.getenv("CONFIG_ENV") or raw_config.get("environment", "local")
    )

    base_cfg = raw_config.get("base", raw_config)
    environments = raw_config.get("environments", {})
    selected_env = environments.get(env_name, {})

    merged = _deep_merge(dict(base_cfg), selected_env)
    merged.setdefault("environment", env_name)

    merged = _expand_and_resolve_paths(merged, config_path.parent)

    return PipelineConfig.model_validate(merged)


__all__ = ["load_config"]
