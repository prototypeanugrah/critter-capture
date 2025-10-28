"""Command line entrypoints for running pipelines."""

from __future__ import annotations

import argparse
from pathlib import Path

from critter_capture.pipelines import (
    run_deployment_pipeline,
    run_inference_pipeline,
    run_training_pipeline,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Critter Capture pipelines")
    parser.add_argument(
        "--pipeline",
        choices=["train", "deploy", "inference"],
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
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.pipeline == "train":
        run_training_pipeline(args.config, args.env)
    elif args.pipeline == "deploy":
        run_deployment_pipeline(args.config, args.env, args.run_id)
    elif args.pipeline == "inference":
        run_inference_pipeline(args.config, args.env)
    else:
        raise ValueError(f"Unknown pipeline {args.pipeline}")


if __name__ == "__main__":  # pragma: no cover
    main()
