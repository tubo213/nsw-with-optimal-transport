from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata

from ._registry import register_optimizer
from .base import BaseOptimizer

__all__ = ["GreedyOptimizer", "greedy"]


def compute_greedy(
    rel_mat: NDArray[np.float_],  # (n_query, n_doc)
    expo: NDArray[np.float_],  # (K, 1)
) -> NDArray[np.float_]:
    n_query, n_doc = rel_mat.shape
    K = expo.shape[0]

    pi = np.zeros((n_query, n_doc, K))
    for k in np.arange(K):
        pi_at_k = np.zeros_like(rel_mat)
        pi_at_k[rankdata(-rel_mat, axis=1, method="ordinal") == k + 1] = 1
        pi[:, :, k] = pi_at_k

    return pi


class GreedyOptimizer(BaseOptimizer):
    def solve(self, rel_mat: NDArray[np.float_], expo: NDArray[np.float_]) -> NDArray[np.float_]:
        return compute_greedy(rel_mat, expo)


@register_optimizer
def greedy(**kwargs: Any) -> GreedyOptimizer:
    return GreedyOptimizer(**kwargs)
