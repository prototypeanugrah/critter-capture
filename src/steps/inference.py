import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from zenml.steps import step

from src.config import DataConfig, InferenceConfig, TrainConfig
from src.data.dataloader import build_transforms
from src.models.resnet18 import AnimalClassifierResNet18

LOGGER = logging.getLogger(__name__)


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


@step(enable_cache=False)
def load_image_for_inference() -> np.ndarray:
    """Load a single image from disk and convert it into a numpy array."""

    config_path = _find_config_file("config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        data_config = DataConfig(**config["data"])
        inference_config = InferenceConfig(**config["inference"])

    image_path = inference_config.input_image_path
    if image_path is None:
        raise ValueError(
            "Set inference.input_image_path in the configuration before running inference."
        )

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Inference image not found at {path}.")

    _, eval_transform = build_transforms(image_size=data_config.image_size)
    image = Image.open(path).convert("RGB")
    tensor = eval_transform(image).unsqueeze(0)
    LOGGER.info("Loaded inference image from %s", path)
    return tensor.numpy()


@step(enable_cache=False)
def predictor(
    image_array: np.ndarray,
) -> Tuple[int, float]:
    """Run an inference request on the model and return the predicted index and confidence."""

    config_path = _find_config_file("config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        data_config = DataConfig(**config["data"])
        train_config = TrainConfig(**config["train"])
        inference_config = InferenceConfig(**config["inference"])

    model_path = Path(inference_config.model_path)
    if model_path is None or not model_path.exists():
        raise ValueError(
            f"Model path not found: {model_path}. "
            "Set inference.model_path in the configuration before running inference."
        )

    # Load the state dict
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))

    # Instantiate the model with the same parameters used during training
    model = AnimalClassifierResNet18(
        num_classes=data_config.num_classes,
        optimizer=train_config.optimizer,
        pretrained=train_config.pretrained,
        lr=train_config.lr,
        max_lr=train_config.max_lr,
        epochs=train_config.epochs,
        device="cpu",  # Use CPU for inference
        train_loader=None,  # Not needed for inference
        class_weights=None,  # Not needed for inference
    )

    # Load the state dict into the model
    model.model.load_state_dict(state_dict)

    # Convert numpy array to tensor
    image_tensor = torch.from_numpy(image_array).float()

    model.model.eval()
    with torch.no_grad():
        prediction = model.model(image_tensor)

    probabilities = torch.nn.functional.softmax(prediction, dim=1)
    probabilities = probabilities[0]
    predicted_index = int(np.argmax(probabilities))

    confidence = float(probabilities[predicted_index])
    LOGGER.info("Predicted index: %s", predicted_index)
    LOGGER.info("Confidence: %s", confidence)
    return predicted_index, confidence
