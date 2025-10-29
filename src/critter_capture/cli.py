"""Command line entrypoints for running pipelines."""

from __future__ import annotations

import argparse
from pathlib import Path

from critter_capture.pipelines import (
    run_deployment_pipeline,
    run_inference_pipeline,
    run_training_pipeline,
)
from critter_capture.zenml import (
    run_deployment_pipeline_with_zenml,
    run_inference_pipeline_with_zenml,
    run_training_pipeline_with_zenml,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Critter Capture pipelines")
    parser.add_argument(
        "--pipeline",
        choices=["train", "deploy", "inference", "validate"],
        required=True,
        default="train",
        help="Pipeline to run",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/pipeline.yaml"),
        help="Path to pipeline configuration file.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Configuration environment override (e.g., local, ci, prod).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID to load training result from.",
    )
    parser.add_argument(
        "--use-zenml",
        action="store_true",
        help="Execute the pipeline through the ZenML orchestrator.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config_path = Path(args.config)

    if args.pipeline == "train":
        if args.use_zenml:
            run_training_pipeline_with_zenml(config_path, args.env)
        else:
            run_training_pipeline(config_path, args.env)
    elif args.pipeline == "deploy":
        if args.use_zenml:
            run_deployment_pipeline_with_zenml(config_path, args.env, args.run_id)
        else:
            run_deployment_pipeline(config_path, args.env, args.run_id)
    elif args.pipeline == "inference":
        if args.use_zenml:
            run_inference_pipeline_with_zenml(config_path, args.env)
        else:
            run_inference_pipeline(config_path, args.env)
    else:
        raise ValueError(f"Unknown pipeline {args.pipeline}")


if __name__ == "__main__":  # pragma: no cover
    main()
