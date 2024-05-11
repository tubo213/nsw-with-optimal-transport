import numpy as np
from numpy.typing import NDArray

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
    # piの制約を満たしているかチェック
    validate_pi(pi)

    n_query, n_doc = rel_mat.shape
    expo_mat = (pi * v.T).sum(2)
    click_mat: NDArray[np.float_] = rel_mat * expo_mat
    user_util = click_mat.sum() / n_query
    item_utils: NDArray[np.float_] = click_mat.sum(0) / n_query
    item_utils_unif = compute_item_utils_unif(rel_mat, v)
    nsw: float = np.power(item_utils.prod(), 1 / n_doc)

    # mean max envy
    max_envies = np.zeros(n_doc)
    for i in range(n_doc):
        u_d_swap = (expo_mat * rel_mat[:, [i] * n_doc]).sum(0)
        d_envies = u_d_swap - u_d_swap[i]
        max_envies[i] = d_envies.max() / n_query
    mean_max_envy = max_envies.mean()

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
