from dataclasses import dataclass
from typing import Any, Literal

__all__ = ["Config", "OptimizerConfig", "GeneratorConfig"]


@dataclass
class OptimizerConfig:
    name: str
    params: dict[str, Any]


@dataclass
class GeneratorConfig:
    name: str
    params: dict[str, Any]
    K: int
    shape: Literal["inv", "exp"]


@dataclass
class Config:
    exp_name: str
    seed: int
    optimizer: OptimizerConfig
    generator: GeneratorConfig
