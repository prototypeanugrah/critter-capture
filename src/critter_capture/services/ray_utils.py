"""
Helper utilities to interact with Ray Tune.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

LOGGER = logging.getLogger(__name__)


def init_ray(ignore_reinit_error: bool = True) -> None:
    if ray.is_initialized():
        LOGGER.debug("Ray is already initialized.")
        return
    ray.init(
        ignore_reinit_error=ignore_reinit_error,
        include_dashboard=False,
        log_to_driver=False,
    )
    LOGGER.info("Initialized Ray runtime.")


def shutdown_ray() -> None:
    if ray.is_initialized():
        ray.shutdown()
        LOGGER.info("Shut down Ray runtime.")


def build_scheduler(max_epochs: int, grace_period: int, metric: str, mode: str) -> ASHAScheduler:
    return ASHAScheduler(
        max_t=max_epochs,
        grace_period=grace_period,
        reduction_factor=2,
        time_attr="training_iteration",
        metric=metric,
        mode=mode,
    )


def run_tune(
    trainable: Callable[[Dict[str, Any]], None],
    search_space: Dict[str, Any],
    num_samples: int,
    scheduler: ASHAScheduler,
    resources_per_trial: Dict[str, Any],
    metric: str,
    mode: str,
) -> tune.ResultGrid:
    """Execute a Ray Tune hyperparameter search."""

    parameter_space = {}
    for key, definitions in search_space.items():
        if "loguniform" in definitions:
            low, high = definitions["loguniform"]
            parameter_space[key] = tune.loguniform(low, high)
        elif "uniform" in definitions:
            low, high = definitions["uniform"]
            parameter_space[key] = tune.uniform(low, high)
        elif "choice" in definitions:
            parameter_space[key] = tune.choice(definitions["choice"])
        else:
            raise ValueError(
                f"Unsupported search space definition for {key}: {definitions}"
            )

    tune_config_kwargs: Dict[str, Any] = {
        "scheduler": scheduler,
        "num_samples": num_samples,
    }

    scheduler_metric = getattr(scheduler, "metric", None)
    scheduler_mode = getattr(scheduler, "mode", None)
    if not scheduler_metric and not scheduler_mode:
        tune_config_kwargs.update({"metric": metric, "mode": mode})

    tuner = tune.Tuner(
        tune.with_resources(trainable, resources=resources_per_trial),
        tune_config=tune.TuneConfig(**tune_config_kwargs),
        param_space=parameter_space,
    )

    LOGGER.info("Launching Ray Tune with %d samples.", num_samples)
    return tuner.fit()


__all__ = ["init_ray", "shutdown_ray", "build_scheduler", "run_tune"]
