from typing import Tuple

from zenml import pipeline

from src.steps.inference import (
    load_image_for_inference,
    predictor,
)


@pipeline(enable_cache=False)
def inference_pipeline() -> Tuple[int, float]:
    """Inference pipeline that loads an image and makes a prediction."""

    image_array = load_image_for_inference()
    predicted_index, confidence = predictor(
        image_array=image_array,
    )
    return predicted_index, confidence
