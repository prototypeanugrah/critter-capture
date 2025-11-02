# Animal Classifier MLOps Pipeline

## Overview
This repository contains an end-to-end workflow for training, evaluating, deploying, and testing a deep-learning animal image classifier. The data comes from an iNaturalist export (birds and mammals) and the model is a configurable ResNet that is trained with PyTorch. ZenML orchestrates the pipelines, MLflow tracks every run and hosts deployed models, and Streamlit provides an interactive UI for manual testing.

## Architecture and MLOps Foundations
- `ZenML` manages reproducible pipelines for data preparation, training, evaluation, deployment, and inference.
- `MLflow` captures parameters, metrics, and artifacts for every pipeline step and exposes the deployed model through the MLflow model deployer.
- Configuration-driven workflows (`src/steps/config.yaml`) make experiments repeatable and allow the pipelines to update metadata such as the discovered class count.
- Automation scripts under `scripts/` provision a local ZenML stack and run pipelines for smoke testing.
- Streamlit consumes the latest MLflow model to let stakeholders validate predictions outside of the pipeline runs.

## Repository Structure
- `data/` raw exports, filtered CSVs, and cached images from iNaturalist.
- `models/` checkpoint artifacts written by the training and deployment pipelines.
- `mlruns/` MLflow tracking store created by ZenML (metrics, params, artifacts, registered models).
- `scripts/` helper bash scripts for training, deployment, and inference smoke tests.
- `src/` configuration schemas, pipeline definitions, ZenML steps, model code, and utilities.
- `run_training.py` entry point for the training pipeline.
- `run_deployment.py` entry point for the deployment pipeline.
- `run_inference.py` entry point for the inference/validation pipeline.
- `streamlit_app.py` Streamlit application that loads the latest deployed model for manual predictions.
- `launch_mlflow_ui.py` convenience script to open the MLflow UI pointed at the local tracking store.

## Setup
### Prerequisites
- macOS or Linux with Python 3.12+
- Git and `curl`

Commands below assume the repository root as the working directory.

### Install dependencies with `uv`
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
uv venv
source .venv/bin/activate
uv sync
```

## Pipelines
### Training pipeline
Defined at `src/pipelines/train_pipeline.py`, this ZenML pipeline wires together three steps: `load_data`, `train_model`, and `evaluate_model`.
1. `load_data` reads `config.yaml` (defaults to `src/steps/config.yaml`), downloads or reuses cached images, and populates the train/val/test `DataLoader`s while updating `data.num_classes`.
2. `train_model` trains the configured ResNet (`src/models/resnet18.py`), writes the best checkpoint to `models/best_model.pth`, logs hyperparameters/metrics/model to MLflow, and records the registered model version plus checkpoint path back into the config under `inference.*`.
3. `evaluate_model` scores the test split, logs accuracy/precision/recall/F1 to MLflow, and returns the metrics to the pipeline context.

Run the pipeline straight from the repo root:
```bash
uv run run_training.py
```
Customize hyperparameters or dataset paths by editing a copy of `src/steps/config.yaml` before execution.

### Deployment pipeline
Trains and evaluates the model, passes the metrics through the `deployment_trigger`, and, when thresholds are satisfied, registers and serves the model through the MLflow Model Deployer. The resulting REST endpoint URL is printed at the end of the run.
```bash
uv run run_deployment.py
```
The helper `scripts/deploy.sh` performs the same flow after ensuring the ZenML stack and MLflow tracker exist.

### Inference pipeline
The latest inference pipeline (`src/pipelines/inference_pipeline.py`) performs offline scoring with the checkpoint produced during training.
1. `load_image_for_inference` pulls the image path from `inference.input_image_path`, applies the same transforms as training, and produces a batched tensor.
2. `predictor` loads the saved `models/best_model.pth` (path injected by the training step), restores the network weights, and returns the predicted class index with its confidence score.

Set the image path in your config, then launch:
```bash
uv run run_inference.py
```
Make sure `inference.input_image_path` points to the file you want to score. The predicted index and probability are captured as ZenML artifacts for the pipeline run, making this a quick regression or smoke test against the latest trained weights. This offline flow complements the online service exposed by the deployment pipeline.

### Streamlit evaluation
Launch the UI for ad-hoc testing once a model version has been registered:
```bash
streamlit run streamlit_app.py
```
Upload an image to compare predicted labels and confidences. The app fetches the latest MLflow model version and reuses the same preprocessing transforms defined in the pipeline.

## Experiment Tracking with MLflow
- The training and deployment pipelines attach the `mlflow_tracker`, so hyperparameters, metrics, confusion matrices, and trained weights are versioned automatically.
- Inspect runs through the ZenML dashboard link printed after `zenml login --local` or launch the UI manually:
  ```bash
  uv run mlflow ui --backend-store-uri file:./mlruns --port 5000
  ```
- `launch_mlflow_ui.py` is a shortcut that uses the current tracking URI and opens the UI on port `4997`.

### Configure ZenML and the local MLflow stack
ZenML needs a local server, experiment tracker, and model deployer:
```bash
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES  # required on macOS for multiprocessing
uv run zenml login --local
mkdir -p mlruns
uv run zenml experiment-tracker register mlflow_tracker --flavor=mlflow --tracking_uri='file:./mlruns'
uv run zenml model-deployer register mlflow --flavor=mlflow
uv run zenml stack register local_mlflow_stack -o default -a default -e mlflow_tracker -d mlflow --set
uv run zenml stack describe
```
If port `8237` is busy, rerun `zenml login --local --port 0`. Scripted alternatives (`scripts/train.sh` or `scripts/deploy.sh`) perform the same setup and run the corresponding pipeline.

### Resetting the stack (optional)
Log out and remove ZenML caches if you need a clean slate:
```bash
uv run zenml logout --local
rm -rf ~/.config/zenml .zen mlruns
uv run zenml status
```
Then repeat the setup commands above.

## Troubleshooting
- `zenml stack describe` should list `mlflow_tracker`; register or set the stack again if it does not.
- The configuration defaults to CUDA. If you are on CPU-only hardware, set `train.device=cpu` in your config before running pipelines.
- The first data prep pass may download images from iNaturalist; ensure you have network access and that `data/raw/images` is writable.

With the stack configured, you can iterate on models via the training pipeline, gate releases with the deployment pipeline, monitor experiments in MLflow, and validate predictions either through the inference pipeline or the Streamlit app.
