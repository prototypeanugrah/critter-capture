"""Streamlit application for the Animal Species classifier."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import streamlit as st
import torch
from PIL import Image

from critter_capture.config import load_config
from critter_capture.data.transforms import build_transforms
from critter_capture.services import upload_feedback
from critter_capture.utils.env import load_env_file


@st.cache_resource
def get_config(
    path: str = "config/pipeline.yaml",
    environment: str | None = None,
):
    return load_config(Path(path), environment)


@st.cache_resource
def get_label_names() -> List[str] | None:
    metadata_path = Path("outputs/metadata.json")
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            labels = metadata.get("label_names")
            if isinstance(labels, list):
                return labels
        except json.JSONDecodeError:
            return None
    return None


@st.cache_resource
def get_transforms(image_size: int):
    _, eval_transform = build_transforms(
        image_size=image_size,
        augment=False,
    )
    return eval_transform


async def request_prediction(
    url: str,
    tensor: torch.Tensor,
) -> List[List[float]]:
    import httpx

    async with httpx.AsyncClient(timeout=30.0) as client:
        payload = {"instances": tensor.unsqueeze(0).numpy().tolist()}
        response = await client.post(url, json=payload)
        response.raise_for_status()
        content = response.json()
        return content["predictions"]


def main() -> None:
    load_env_file()
    st.title("Animal Species Classifier")
    st.write("Upload a wildlife photo to receive species probabilities.")

    cfg = get_config()
    transform = get_transforms(cfg.data.image_size)
    service_url = f"http://{cfg.deployment.serving_host}:{cfg.deployment.serving_port}/invocations"

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_column_width=True)

        tensor = transform(image)

        with st.spinner("Querying model..."):
            try:
                predictions = asyncio.run(
                    request_prediction(
                        service_url,
                        tensor,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to fetch prediction: {exc}")
                st.stop()

        scores = np.array(predictions[0])
        metadata_labels = get_label_names()
        if metadata_labels and len(metadata_labels) == len(scores):
            labels = metadata_labels
        else:
            labels = [f"Class {idx}" for idx in range(len(scores))]

        sorted_indices = np.argsort(scores)[::-1]
        st.subheader("Prediction Probabilities")
        prob_rows = {"Label": [], "Probability": []}
        for idx in sorted_indices[:5]:
            prob_rows["Label"].append(labels[idx])
            prob_rows["Probability"].append(float(scores[idx]))
        st.table(prob_rows)

        predicted_label = labels[sorted_indices[0]]
        st.markdown(
            f"**Top prediction:** {predicted_label} (p={scores[sorted_indices[0]]:.3f})"
        )

        st.subheader("Feedback")
        correct_label = st.selectbox(
            "Select the correct label",
            options=labels,
            index=sorted_indices[0],
        )
        feedback_quality = st.selectbox(
            "How accurate was the prediction?",
            ["great", "good", "fair", "poor"],
        )
        submit = st.button("Submit Feedback")

        if submit:
            try:
                temp_dir = Path(tempfile.mkdtemp())
                image_path = temp_dir / uploaded_file.name
                image.save(image_path)

                metadata = {
                    "predictions": {
                        label: float(prob)
                        for label, prob in zip(
                            labels,
                            scores,
                        )
                    },
                    "predicted_label": predicted_label,
                    "selected_label": correct_label,
                    "feedback_quality": feedback_quality,
                    "model_endpoint": service_url,
                }

                upload_feedback(
                    bucket=cfg.storage.s3_bucket,
                    region=cfg.storage.s3_region,
                    key_prefix="feedback",
                    image_path=image_path,
                    metadata=metadata,
                )
                st.success("Feedback submitted. Thank you!")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to submit feedback: {exc}")


if __name__ == "__main__":
    main()
