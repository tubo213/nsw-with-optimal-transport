from ._registry import optimizer_entrypoint
from .base import BaseOptimizer

__all__ = ["create_optimizer"]


# timmを参考にしたFactory Methodパターンでoptimizerを作成
# ref: https://zenn.dev/ycarbon/articles/75c9c71e0c2df3
def create_optimizer(
    optimizer_name: str,
    **kwargs,
) -> BaseOptimizer:
    create_fn = optimizer_entrypoint(optimizer_name)
    return create_fn(**kwargs)
