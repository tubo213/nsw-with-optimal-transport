from typing import Any

from ._registry import generator_entrypoint
from .base import BaseGenerator

__all__ = ["create_generator"]


# timmを参考にしたFactory Methodパターンでoptimizerを作成
# ref: https://zenn.dev/ycarbon/articles/75c9c71e0c2df3
def create_generator(
    generator_name: str,
    **kwargs: Any,
) -> BaseGenerator:
    create_fn = generator_entrypoint(generator_name)
    return create_fn(**kwargs)
