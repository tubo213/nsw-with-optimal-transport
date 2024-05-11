from typing import Any, Optional

import cvxpy as cvx
import numpy as np
from numpy.typing import NDArray

from ._registry import register_optimizer
from .base import BaseClusteredOptimizer, BaseOptimizer

__all__ = ["NSWOptimizer", "ClusteredNSWOptimizer", "nsw", "clustered_nsw"]


def compute_nsw(
    rel_mat: NDArray[np.float_],
    expo: NDArray[np.float_],
    high: NDArray[np.float_],
    alpha: float = 0,
    solver: Optional[str] = None,
) -> NDArray[np.float_]:
    """_summary_

    Args:
        rel_mat (NDArray[np.float_]): _description_
        expo (NDArray[np.float_]): _description_
        high (NDArray[np.float_]): _description_
        alpha (float, optional): _description_. Defaults to 0.
        solver (Optional[str], optional): _description_. Defaults to None.

    Returns:
        NDArray[np.float_]: _description_

    References:
        - This function is adapted from `compute_pi_nsw` at [https://github.com/usaito/kdd2022-fair-ranking-nsw/blob/main/src/synthetic/func.py] by yuta-saito.
        - MIT License, Copyright (c) 2022 yuta-saito. See the original repository for full license information.
    """
    n_query, n_doc = rel_mat.shape
    K = expo.shape[0]
    query_basis = np.ones((n_query, 1))
    am_rel = rel_mat.sum(0) ** alpha

    pi = cvx.Variable((n_query, n_doc * K))
    obj = 0.0
    constraints = []
    for d in np.arange(n_doc):
        obj += am_rel[d] * cvx.log(rel_mat[:, d] @ pi[:, K * d : K * (d + 1)] @ expo)  # type: ignore
        # feasible allocation
        basis_ = np.zeros((n_doc * K, 1))
        basis_[K * d : K * (d + 1)] = 1
        constraints += [pi @ basis_ <= query_basis * high[d]]
    # feasible allocation
    for k in np.arange(K):
        basis_ = np.zeros((n_doc * K, 1))
        basis_[np.arange(n_doc) * K + k] = 1
        constraints += [pi @ basis_ <= query_basis]
    constraints += [pi <= 1.0]
    constraints += [0.0 <= pi]

    prob = cvx.Problem(cvx.Maximize(obj), constraints)
    prob.solve(solver=solver, verbose=False)
    pi_arr: NDArray[np.float_] = pi.value.reshape((n_query, n_doc, K))
    pi_arr = np.clip(pi_arr, 0.0, 1.0)

    return pi_arr


class NSWOptimizer(BaseOptimizer):
    def __init__(self, alpha: float = 0, solver: Optional[str] = None):
        self.alpha = alpha
        self.solver = solver

    def solve(self, rel_mat: NDArray[np.float_], expo: NDArray[np.float_]) -> NDArray[np.float_]:
        n_doc = rel_mat.shape[1]
        high = np.ones(n_doc)
        return compute_nsw(rel_mat, expo, high, self.alpha, solver=self.solver)


class ClusteredNSWOptimizer(BaseClusteredOptimizer):
    def __init__(
        self,
        n_doc_cluster: int,
        n_query_cluster: int,
        alpha: float = 0,
        solver: Optional[str] = None,
        random_state: int = 12345,
    ):
        super().__init__(n_doc_cluster, n_query_cluster, random_state)
        self.alpha = alpha
        self.solver = solver

    def _solve(
        self, rel_mat: NDArray[np.float_], expo: NDArray[np.float_], high: NDArray[np.float_]
    ) -> NDArray[np.float_]:
        return compute_nsw(rel_mat, expo, high, self.alpha, solver=self.solver)


@register_optimizer
def nsw(**kwargs: Any) -> NSWOptimizer:
    return NSWOptimizer(**kwargs)


@register_optimizer
def clustered_nsw(**kwargs: Any) -> ClusteredNSWOptimizer:
    return ClusteredNSWOptimizer(**kwargs)
