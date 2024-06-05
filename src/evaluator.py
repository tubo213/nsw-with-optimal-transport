import numpy as np
import torch
from numpy.typing import NDArray
from tqdm import tqdm

__all__ = ["evaluate_pi"]


def compute_item_utils_unif(
    rel_mat: NDArray[np.float_],
    v: NDArray[np.float_],
) -> NDArray[np.float_]:
    n_query, n_doc = rel_mat.shape
    K = v.shape[0]
    unif_pi = np.ones((n_query, n_doc, K)) / n_doc
    expo_mat = (unif_pi * v.T).sum(2)
    click_mat: NDArray[np.float_] = rel_mat * expo_mat
    item_utils: NDArray[np.float_] = click_mat.sum(0) / n_query

    return item_utils


# ref: https://github.com/usaito/kdd2022-fair-ranking-nsw
def evaluate_pi(
    pi: NDArray[np.float_], rel_mat: NDArray[np.float_], v: NDArray[np.float_]
) -> dict[str, float]:
    """Evaluate the given probability matrix pi.

    Args:
        pi (NDArray[np.float_]): The probability matrix to be evaluated. (n_query, n_doc, n_rank)
        rel_mat (NDArray[np.float_]): The true relevance matrix. (n_query, n_doc)
        v (NDArray[np.float_]): The exposure vector. (n_rank, 1)

    Returns:
        dict[str, float]: A dictionary containing the evaluation metrics.

    References:
        - This function is adapted from `evaluate_pi` at [https://github.com/usaito/kdd2022-fair-ranking-nsw/blob/main/src/synthetic/func.py] by yuta-saito.
        - MIT License, Copyright (c) 2022 yuta-saito. See the original repository for full license information.
    """
    # piの制約を満たしているかチェック
    validate_pi(pi)

    n_query, n_doc = rel_mat.shape
    expo_mat = (pi * v.T).sum(2)
    click_mat: NDArray[np.float_] = rel_mat * expo_mat
    user_util = click_mat.sum() / n_query
    item_utils: NDArray[np.float_] = click_mat.sum(0) / n_query
    item_utils_unif = compute_item_utils_unif(rel_mat, v)
    nsw: float = np.power(item_utils.prod(), 1 / n_doc)

    # torchでmean max envyを計算
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expo_mat_tensor = torch.tensor(expo_mat).to(device)
    rel_mat_tensor = torch.tensor(rel_mat).to(device)
    n_query_tensor = torch.tensor(n_query).to(device)
    max_envies_tensor = torch.zeros(n_doc).to(device)
    for i in tqdm(range(n_doc), desc="Computing mean max envy", leave=False):
        u_d_swap = (expo_mat_tensor * rel_mat_tensor[:, [i] * n_doc]).sum(0)
        d_envies = u_d_swap - u_d_swap[i]
        max_envies_tensor[i] = d_envies.max() / n_query_tensor
    mean_max_envy = max_envies_tensor.mean().item()

    # pct item util better off
    pct_item_util_better = 100 * ((item_utils / item_utils_unif) > 1.10).mean()
    # pct item util worse off
    pct_item_util_worse = 100 * ((item_utils / item_utils_unif) < 0.90).mean()

    return {
        "nsw": nsw,
        "user_util": user_util,
        "mean_max_envy": mean_max_envy,
        "pct_item_util_better": pct_item_util_better,
        "pct_item_util_worse": pct_item_util_worse,
    }


def validate_pi(pi: NDArray[np.float_], eps: float = 1e-4) -> None:
    """Validate the probability matrix pi.

    This function checks if the given probability matrix pi satisfies certain conditions.

    Args:
        pi (NDArray[np.float_]): The probability matrix to be validated. (n_query, n_doc, n_rank)
        eps (float, optional): The tolerance for floating-point comparisons. Defaults to 1e-5.

    Raises:
        AssertionError: If any of the validation conditions are not met.
    """
    # piにnanが含まれていない
    assert not np.isnan(pi).any(), "piにnanが含まれています"

    # 0 <= pi_{uik} <= 1 \forall (u, i, k), piの全ての要素が0以上1以下
    assert np.all((0 <= pi) & (pi <= 1)), "piの全ての要素が0以上1以下でないです"

    # \sum_{i} \sum_{k} pi_{uik} = n_rank \forall u, ランキングの合計値が表示数と一致
    n_rank = pi.shape[2]
    assert np.all(
        np.abs(pi.sum((1, 2)) - n_rank) < eps
    ), "ランキングの合計値が表示数と一致しません"

    # \sum_{i} pi_{uik} = 1 \forall (u, k), ランキングへの流入量が1
    assert np.all(np.abs(pi.sum(1) - 1) < eps), "ランキングへの流入量が1でないです"
