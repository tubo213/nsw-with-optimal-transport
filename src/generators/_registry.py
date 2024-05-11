import warnings
from typing import Callable

from .base import BaseGenerator

__all__ = ["register_generator", "list_generators"]

_generator_entrypoints = {}


def register_generator(fn: Callable[..., BaseGenerator]) -> Callable[..., BaseGenerator]:
    generator_name = fn.__name__

    if generator_name in _generator_entrypoints:
        warnings.warn(f"Duplicate generator {generator_name}", stacklevel=2)

    _generator_entrypoints[generator_name] = fn

    return fn


def generator_entrypoint(generator_name: str) -> Callable[..., BaseGenerator]:
    available_generators = list_generators()
    if generator_name not in available_generators:
        raise ValueError(
            f"Invalid generator name: {generator_name}, available generators are {available_generators}"  # noqa
        )

    return _generator_entrypoints[generator_name]


def list_generators() -> list[str]:
    generators = list(_generator_entrypoints.keys())
    return sorted(generators)
