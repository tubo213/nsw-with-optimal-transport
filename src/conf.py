from dataclasses import dataclass
from typing import Literal

__all__ = ["Config", "OptimizerConfig", "GeneratorConfig"]


@dataclass
class OptimizerConfig:
    name: str
    params: dict


@dataclass
class GeneratorConfig:
    n_query: int
    n_doc: int
    lam: float
    flip_ratio: float
    noise: float
    K: int
    shape: Literal["inv", "exp", "log"]


@dataclass
class Config:
    exp_name: str
    seed: int
    optimizer: OptimizerConfig
    generator: GeneratorConfig
