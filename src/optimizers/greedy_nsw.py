from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._registry import register_optimizer
from .base import BaseOptimizer

__all__ = ["GreedyNSWOptimizer", "greedy_nsw"]


def choice_maximize_nsw_doc(
    impact_on_items: NDArray[np.float_],  # (n_doc)
    impact_qk: NDArray[np.float_],  # (n_doc)
    used_docs: NDArray[np.float_],  # (n_doc)
    eps: float = 1e-5,
) -> np.intp:
    # 既に選択されたdocの影響は0
    impact_qk = impact_qk * ~used_docs  # (n_doc)

    # 各docを選択した場合の影響を計算
    impact_qk_diag = np.diag(impact_qk)  # (n_doc, n_doc)
    this_impact_on_items = impact_on_items[:, None] + impact_qk_diag  # (n_doc, n_doc)

    # nswの目的関数値が一番大きくなるdocを選択
    ln_nsw = np.log(this_impact_on_items + eps).sum(0)  # (n_doc)
    best_doc = np.argmax(ln_nsw)

    return best_doc


def compute_greedy_nsw(
    rel_mat: NDArray[np.float_],  # (n_query, n_doc)
    expo: NDArray[np.float_],  # (K, 1)
) -> NDArray[np.float_]:
    n_query, n_doc = rel_mat.shape
    K = expo.shape[0]
    impact = rel_mat[:, :, None] * expo.reshape(1, 1, K)  # (n_query, n_doc, K)
    impact_on_items = np.zeros(n_doc)  # (n_doc)
    pi = np.zeros((n_query, n_doc, K))  # (n_query, n_doc, K)
    used_docs = np.zeros((n_query, n_doc), dtype=bool)  # (n_query, n_doc)

    for k in range(K):
        for q in range(n_query):
            impact_qk = impact[q, :, k]  # (n_doc)
            # 目的関数が最も大きくなるdocを選択
            best_doc = choice_maximize_nsw_doc(impact_on_items, impact_qk, used_docs[q])

            # 更新
            impact_on_items[best_doc] += impact_qk[best_doc]
            pi[q, best_doc, k] = 1
            used_docs[q, best_doc] = True

    assert np.all(pi.sum(1) == 1), "pi.sum(1) should be equal to 1"
    assert np.all(pi.sum(2) <= 1), "pi.sum(2) should be less than or equal to 1"

    return pi


class GreedyNSWOptimizer(BaseOptimizer):
    def solve(self, rel_mat: NDArray[np.float_], expo: NDArray[np.float_]) -> NDArray[np.float_]:
        return compute_greedy_nsw(rel_mat, expo)


@register_optimizer
def greedy_nsw(**kwargs: Any) -> GreedyNSWOptimizer:
    return GreedyNSWOptimizer(**kwargs)
