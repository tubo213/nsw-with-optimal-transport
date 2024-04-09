import warnings
from typing import Callable

from .base import BaseOptimizer

__all__ = ["register_optimizer", "list_optimizers"]

_optimizer_entrypoints = {}


def register_optimizer(fn: Callable[..., BaseOptimizer]) -> Callable[..., BaseOptimizer]:
    optimizer_name = fn.__name__

    if optimizer_name in _optimizer_entrypoints:
        warnings.warn(f"Duplicate optimizer {optimizer_name}", stacklevel=2)

    _optimizer_entrypoints[optimizer_name] = fn

    return fn


def optimizer_entrypoint(optimizer_name: str) -> Callable[..., BaseOptimizer]:
    available_optimizers = list_optimizers()
    if optimizer_name not in available_optimizers:
        raise ValueError(
            f"Invalid optimizer name: {optimizer_name}, available optimizers are {available_optimizers}"  # noqa
        )

    return _optimizer_entrypoints[optimizer_name]


def list_optimizers() -> list[str]:
    optimizers = list(_optimizer_entrypoints.keys())
    return sorted(optimizers)
