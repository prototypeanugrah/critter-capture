from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import streamlit as st
import torch
import yaml
from PIL import Image
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)

from src.config import DataConfig, InferenceConfig, TrainConfig
from src.data.dataloader import build_transforms

LOGGER = logging.getLogger(__name__)


@st.cache_resource
def _find_config_file(filename: str) -> Path:
    """Find the config.yaml file in common locations."""
    # Try current directory first (for temp files from deploy script)
    if Path(filename).exists():
        return Path(filename)

    # Try project root
    project_root = Path(__file__).parent.parent.parent
    if (project_root / filename).exists():
        return project_root / filename

    # Try src/steps/config.yaml
    if (project_root / "src" / "steps" / filename).exists():
        return project_root / "src" / "steps" / filename

    # Default fallback
    return Path(filename)


def load_label_names(data_config) -> Tuple[List[int], Dict[int, str]]:
    if not Path(data_config.csv_path).exists():
        raise FileNotFoundError(
            f"Dataset CSV not found at '{data_config.csv_path}'. Update data.csv_path in config."
        )

    df = pd.read_csv(
        data_config.csv_path,
        usecols=[data_config.label_column, data_config.label_names_column],
    ).dropna()

    mapping = (
        df.drop_duplicates(subset=[data_config.label_column])
        .set_index(data_config.label_column)[data_config.label_names_column]
        .to_dict()
    )
    sorted_ids = sorted(mapping.keys())
    return sorted_ids, mapping


def prepare_image(image: Image.Image, image_size: int) -> np.ndarray:
    _, eval_transform = build_transforms(image_size=image_size)
    tensor = eval_transform(image).unsqueeze(0)
    return tensor.numpy()


def fetch_prediction_service(
    pipeline_name: str,
    step_name: str,
    running: bool,
    wait_seconds: int,
):
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()
    services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=step_name,
        running=running,
    )
    if not services:
        return None

    service = services[0]
    try:
        service.start(timeout=wait_seconds)
    except Exception:  # pragma: no cover - already running
        LOGGER.debug("Service already running.")

    try:
        if hasattr(service, "wait_for_service_ready"):
            service.wait_for_service_ready(timeout=wait_seconds)
        else:
            service.wait(timeout_seconds=wait_seconds)
    except Exception as exc:  # pragma: no cover - external service failure
        LOGGER.warning("Service did not confirm readiness: %s", exc)

    return service


@st.cache_resource
def load_model_directly(
    model_path: Optional[str] = None,
    run_id: Optional[str] = None,
    artifact_path: Optional[str] = None,
) -> torch.nn.Module:
    """
    Load MLflow model directly using run_id or model_path.

    Args:
        model_path: MLflow model URI (e.g., "models:/animal-classifier-resnet50/latest")
                    or path to model directory
        run_id: MLflow run ID (requires artifact_path)
        artifact_path: Artifact path within the run (e.g., "animal-classifier-resnet50")

    Returns:
        Loaded PyTorch model
    """
    if model_path:
        model_uri = model_path
    elif not model_path and not run_id and artifact_path:
        model_uri = artifact_path
    elif run_id and artifact_path:
        model_uri = f"runs:/{run_id}/{artifact_path}"
    else:
        raise ValueError(
            "Either model_path or (run_id + artifact_path) must be provided"
        )

    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    return model


def predict_with_model(
    model: torch.nn.Module,
    image_array: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Run prediction using a directly loaded model."""
    model.to(device)
    with torch.no_grad():
        tensor = torch.from_numpy(image_array).to(device)
        outputs = model(tensor)
        predictions = torch.nn.functional.softmax(outputs, dim=1)
    return predictions.cpu().numpy()


def main() -> None:
    st.set_page_config(page_title="Animal Classifier", layout="centered")
    st.title("🐾 Animal Classifier Demo")
    st.caption(
        "Upload an image and get predictions from the latest deployed MLflow model."
    )

    config_file = "config.yaml"
    config_path = _find_config_file(config_file)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        data_config = DataConfig(**config["data"])
        train_config = TrainConfig(**config["train"])
        inference_config = InferenceConfig(**config["inference"])

    try:
        label_ids, id_to_name = load_label_names(data_config)
    except Exception as exc:
        st.error(f"Unable to load label names: {exc}")
        return

    uploaded_file = st.file_uploader(
        "Select an image",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("Upload an image to run inference.")
        return

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as exc:
        st.error(f"Unable to open the uploaded image: {exc}")
        return

    model_uri = f"mlruns/models/{train_config.mlflow_model_name}/version-{inference_config.model_version}"
    with open(model_uri + "/meta.yaml", "r", encoding="utf-8") as f:
        model_meta = yaml.safe_load(f)

    # run_id = model_meta["run_id"]
    artifact_path = model_meta[
        "storage_location"
    ]  # looks like this: file:///home/public/avaishna2/animal-classifier-mlops/mlruns/484885874116169817/models/m-0a1e1eb16b79427a9aa60dbdff851285/artifacts
    artifact_path = artifact_path.split("file://")[1]
    # model_path = artifact_path + "/" + "data/model.pth"

    st.image(image, caption="Uploaded Image")

    image_array = prepare_image(image, data_config.image_size)

    # Load model directly
    try:
        model = load_model_directly(
            # model_path=model_path,
            # run_id=run_id,
            artifact_path=artifact_path,
        )
        predictions = predict_with_model(model, image_array)
    except Exception as exc:
        st.error(f"Failed to load model or make prediction: {exc}")
        predictions = None
        return

    if predictions.ndim != 2 or predictions.shape[1] == 0:
        st.error("Prediction response has unexpected shape.")
        return

    probabilities = predictions[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_label_id = (
        label_ids[predicted_index]
        if predicted_index < len(label_ids)
        else predicted_index
    )
    predicted_label = id_to_name.get(predicted_label_id, str(predicted_label_id))
    confidence = float(probabilities[predicted_index])

    st.success(
        f"Predicted class: **{predicted_label}** ({confidence * 100:.2f}% confidence)"
    )


if __name__ == "__main__":
    main()
