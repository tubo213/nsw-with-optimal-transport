from typing import Optional

import cvxpy as cvx
import numpy as np

from ._registry import register_optimizer
from .base import BaseClusteredOptimizer, BaseOptimizer

__all__ = ["ExpoFairOptimizer", "ClusteredExpoFairOptimizer", "expo_fair", "clustered_expo_fair"]


def compute_pi_expo_fair(
    rel_mat: np.ndarray,  # (n_query, n_doc)
    expo: np.ndarray,  # (K, 1)
    high: np.ndarray,  # (n_doc, 1)
    solver: Optional[str] = None,
) -> np.ndarray:
    n_query, n_doc = rel_mat.shape
    K = expo.shape[0]
    query_basis = np.ones((n_query, 1))
    am_rel = rel_mat.sum(0)[:, np.newaxis]
    am_expo = expo.sum() * n_query * am_rel / rel_mat.sum()

    pi = cvx.Variable((n_query, n_doc * K))
    obj = 0.0
    constraints = []
    for d in np.arange(n_doc):
        pi_d = pi[:, K * d : K * (d + 1)]
        obj += rel_mat[:, d] @ pi_d @ expo
        # feasible allocation
        basis_ = np.zeros((n_doc * K, 1))
        basis_[K * d : K * (d + 1)] = 1
        constraints += [pi @ basis_ <= query_basis * high[d]]
        # amortized exposure
        constraints += [query_basis.T @ pi_d @ expo <= am_expo[d]]
    # feasible allocation
    for k in np.arange(K):
        basis_ = np.zeros((n_doc * K, 1))
        basis_[np.arange(n_doc) * K + k] = 1
        constraints += [pi @ basis_ <= query_basis]
    constraints += [pi <= 1.0]
    constraints += [0.0 <= pi]

    prob = cvx.Problem(cvx.Maximize(obj), constraints)
    prob.solve(solver=solver, verbose=False)

    pi = pi.value.reshape((n_query, n_doc, K))
    pi = np.clip(pi, 0.0, 1.0)

    return pi


class ExpoFairOptimizer(BaseOptimizer):
    def __init__(self, solver: Optional[str] = None):
        self.solver = solver

    def solve(self, rel_mat: np.ndarray, expo: np.ndarray) -> np.ndarray:
        n_doc = rel_mat.shape[1]
        high = np.ones(n_doc)
        return compute_pi_expo_fair(rel_mat, expo, high, solver=self.solver)


class ClusteredExpoFairOptimizer(BaseClusteredOptimizer):
    def __init__(
        self,
        n_doc_cluster: int,
        n_query_cluster: int,
        solver: Optional[str] = None,
        random_state: int = 12345,
    ):
        super().__init__(n_doc_cluster, n_query_cluster, random_state)
        self.solver = solver

    def _solve(self, rel_mat: np.ndarray, expo: np.ndarray, high: np.ndarray) -> np.ndarray:
        return compute_pi_expo_fair(rel_mat, expo, high, solver=self.solver)


@register_optimizer
def expo_fair(**kwargs) -> ExpoFairOptimizer:
    return ExpoFairOptimizer(**kwargs)


@register_optimizer
def clustered_expo_fair(**kwargs) -> ClusteredExpoFairOptimizer:
    return ClusteredExpoFairOptimizer(**kwargs)