## Critter Capture – Animal Species Classifier

This repository implements a full multi-class image classification workflow for the `observations-632017.csv` dataset (derived from iNaturalist exports). The pipeline ingests the local CSV, downloads and caches the referenced imagery, and produces an animal species classifier named *AnimalSpeciesClassifier*. It couples Ray Tune-powered training with MLflow experiment tracking, a gated continuous deployment flow, and a Streamlit-based inference experience.

### Repository Layout

- `config/` – Pipeline configuration files (`pipeline.yaml`) with environment overrides.
- Model architecture (filters, per-block depth, classifier widths) is controlled via `model.*` fields inside `config/pipeline.yaml`; each environment can override these values to swap between lightweight CI models and the full VGG-style backbone.
- `src/critter_capture/` – Application code (data loaders, models, pipelines, services, utilities).
- `app/` – Streamlit application for interactive inference and feedback capture.
- `scripts/` – Operational scripts, including MLflow tracking server startup.
- `tests/` – Lightweight unit tests for critical components.

### Prerequisites

0. Install [uv](https://github.com/astral-sh/uv) (one-time):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

1. Create virtual environment and install dependencies (Python 3.12+):

   ```bash
   uv venv
   source .venv/bin/activate
   uv sync
   ```

2. (Optional) Export AWS credentials if you plan to use the Streamlit feedback workflow:

   ```bash
   export AWS_ACCESS_KEY_ID=... \
          AWS_SECRET_ACCESS_KEY=... \
          AWS_DEFAULT_REGION=us-east-1
   ```

### Pipelines (Docker-First)

All core flows can be launched via Docker Compose. Start the MLflow tracking server in a separate terminal:

```bash
docker compose up mlflow
```

Then run the individual pipelines as needed:

- **Training** – Build model candidates with Ray Tune and log runs to MLflow:

  ```bash
  docker compose run --rm trainer
  ```

- **Deployment** – Reuse the latest training run, apply gating, and (re)deploy the model server:

  ```bash
  docker compose run --rm deployer
  ```

- **Inference Checks** – Exercise the deployed endpoint and log regression metrics:

  ```bash
  docker compose run --rm inference
  ```

Configuration overrides can still be supplied via `--env` arguments in the compose commands (for example, append `--env docker`).

For a local run outside Docker, use uv to execute the pipeline entrypoints, for example:

```bash
uv run python -m critter_capture.cli train --config config/pipeline.yaml --env local
```

### Docker Usage

Container images are provided to standardise the runtime. Ensure Docker is running and then build the images:

```bash
docker compose build
```

Serve the latest Production model and the Streamlit UI:

```bash
docker compose up model-server streamlit
```

The docker-specific configuration profile (`--env docker`) points pipelines to the containerised MLflow services and uses an externally managed serving endpoint. Restart `model-server` after every successful deployment to load the newly promoted version.

### Streamlit Application

Launch the interactive UI (requires a running MLflow model server):

```bash
docker compose up streamlit
```

Upload a wildlife image to view predicted species probabilities. Provide feedback to push the image and annotations to the configured S3 bucket for future re-training and drift monitoring.

### Additional Notes

- Training defaults to CPU execution. Enable GPU by running with the `prod` environment or editing `training.device` in the configuration.
- Hyperparameter tuning can be disabled (`tuning.enabled: false`) for quick iterations.
- All pipeline runs write logs to `logs/pipeline.log` and structured outputs to the `outputs/` directory.

### Testing

Run the sanity checks:

```bash
uv run pytest
```

These tests cover configuration loading and a smoke test of the CNN forward pass.
