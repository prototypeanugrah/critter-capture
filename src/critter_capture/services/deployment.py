"""
Interactions with the MLflow model serving deployment.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

import os
import signal

import httpx

LOGGER = logging.getLogger(__name__)


def start_mlflow_server(model_uri: str, host: str, port: int, env: Optional[Dict[str, str]] = None) -> subprocess.Popen:
    """Start an MLflow model serving process."""

    command = [
        "mlflow",
        "models",
        "serve",
        "-m",
        model_uri,
        "--host",
        host,
        "--port",
        str(port),
        "--no-conda",
    ]
    LOGGER.info("Starting MLflow serving: %s", " ".join(command))
    combined_env = os.environ.copy()
    if env:
        combined_env.update(env)
    process = subprocess.Popen(command, env=combined_env)
    return process


def update_mlflow_deployment(model_uri: str, host: str, port: int, env: Optional[Dict[str, str]] = None) -> subprocess.Popen:
    """Update the running MLflow serving process by restarting it."""

    return start_mlflow_server(model_uri, host, port, env)


def wait_for_healthcheck(url: str, timeout: int = 60) -> bool:
    """Poll the health endpoint of the MLflow server until ready or timeout."""

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = httpx.get(url, timeout=5.0)
            if response.status_code == 200:
                LOGGER.info("MLflow serving healthy at %s", url)
                return True
        except httpx.HTTPError:
            time.sleep(2)
    LOGGER.error("Timed out waiting for MLflow serving at %s", url)
    return False


def score_payload(url: str, inputs: Dict, timeout: float = 30.0) -> Dict:
    """Send an inference request to the MLflow serving endpoint."""

    response = httpx.post(url, json=inputs, timeout=timeout)
    response.raise_for_status()
    return response.json()


def stop_existing_process(metadata_path: Path) -> None:
    """Stop an existing MLflow serving process if metadata exists."""

    if not metadata_path.exists():
        return

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    pid = metadata.get("pid")
    if not pid:
        return
    try:
        LOGGER.info("Stopping existing MLflow serving process with PID %s", pid)
        os.kill(pid, signal.SIGTERM)
    except OSError:
        LOGGER.warning("Failed to stop process %s; it may not be running.", pid)
    metadata_path.unlink(missing_ok=True)


def save_process_metadata(process: subprocess.Popen, metadata_path: Path) -> None:
    """Persist metadata about the serving process."""

    metadata = {
        "pid": process.pid,
        "command": process.args if isinstance(process.args, list) else [process.args],
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


__all__ = [
    "start_mlflow_server",
    "update_mlflow_deployment",
    "wait_for_healthcheck",
    "score_payload",
    "save_process_metadata",
    "stop_existing_process",
]
