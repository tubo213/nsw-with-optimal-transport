import numpy as np

from src.optimizer.base import BaseOptimizer


def compute_greedy(
    rel_mat: np.ndarray,  # (n_query, n_doc)
    expo: np.ndarray,  # (K, 1)
    eps: float = 1e-5,
) -> np.ndarray:
    n_query, n_doc = rel_mat.shape
    K = expo.shape[0]
    impact = rel_mat[:, :, None] * expo.reshape(1, 1, K)  # (n_query, n_doc, K)
    impact_on_items = np.zeros(n_doc)  # (n_doc)
    pi = np.zeros((n_query, n_doc, K))  # (n_query, n_doc, K)
    used_docs = np.zeros((n_query, n_doc), dtype=bool)  # (n_query, n_doc)

    for k in range(K):
        for q in range(n_query):
            # nswの目的関数値が一番大きくなるdocを選ぶ
            impact_qk = impact[q, :, k] * ~used_docs[q]  # (n_doc)
            this_impact_on_items = impact_on_items[:, None] + impact_qk[None, :]  # (n_doc, n_doc)
            ln_nsw = np.log(this_impact_on_items + eps).sum(0)  # (n_doc)
            best_doc = np.argmax(ln_nsw)
            impact_on_items[best_doc] += impact_qk[best_doc]
            pi[q, best_doc, k] = 1
            used_docs[q, best_doc] = True

    assert np.all(pi.sum(1) == 1), "pi.sum(1) should be equal to 1"
    assert np.all(pi.sum(2) <= 1), "pi.sum(2) should be less than or equal to 1"

    return pi


class GreedyOptimizer(BaseOptimizer):
    def solve(self, rel_mat: np.ndarray, expo: np.ndarray) -> np.ndarray:
        return compute_greedy(rel_mat, expo)
