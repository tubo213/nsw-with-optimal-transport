from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Result:
    user_util: float
    item_utils: float
    max_envies: float
    nsw: float
    exec_time: Optional[float] = None


def evaluate_pi(pi: np.ndarray, rel_mat: np.ndarray, v: np.ndarray) -> Result:
    n_query, n_doc = rel_mat.shape
    expo_mat = (pi * v.T).sum(2)
    click_mat: np.ndarray = rel_mat * expo_mat
    user_util = click_mat.sum() / n_query
    item_utils: np.ndarray = click_mat.sum(0) / n_query
    nsw: float = np.power(item_utils.prod(), 1 / n_doc)

    max_envies = np.zeros(n_doc)
    for i in range(n_doc):
        u_d_swap = (expo_mat * rel_mat[:, [i] * n_doc]).sum(0)
        d_envies = u_d_swap - u_d_swap[i]
        max_envies[i] = d_envies.max() / n_query

    return Result(user_util, item_utils.mean(), max_envies.mean(), nsw)
