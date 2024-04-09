import numpy as np

from ._registry import register_optimizer
from .base import BaseOptimizer

__all__ = ["GreedyOptimizer", "greedy"]


def choice_maximize_nsw_doc(
    impact_on_items: np.ndarray,  # (n_doc)
    impact_qk: np.ndarray,  # (n_doc)
    used_docs: np.ndarray,  # (n_doc)
    eps: float = 1e-5,
) -> int:
    # 既に選択されたdocの影響は0
    impact_qk = impact_qk * ~used_docs  # (n_doc)

    # 各docを選択した場合の影響を計算
    impact_qk_diag = np.diag(impact_qk)  # (n_doc, n_doc)
    this_impact_on_items = impact_on_items[:, None] + impact_qk_diag  # (n_doc, n_doc)

    # nswの目的関数値が一番大きくなるdocを選択
    ln_nsw = np.log(this_impact_on_items + eps).sum(0)  # (n_doc)
    best_doc = np.argmax(ln_nsw)

    return best_doc


def compute_greedy(
    rel_mat: np.ndarray,  # (n_query, n_doc)
    expo: np.ndarray,  # (K, 1)
) -> np.ndarray:
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


class GreedyOptimizer(BaseOptimizer):
    def solve(self, rel_mat: np.ndarray, expo: np.ndarray) -> np.ndarray:
        return compute_greedy(rel_mat, expo)


@register_optimizer
def greedy(**kwargs) -> GreedyOptimizer:
    return GreedyOptimizer(**kwargs)
