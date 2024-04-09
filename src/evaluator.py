import numpy as np

__all__ = ["evaluate_pi"]


def compute_item_utils_unif(
    rel_mat: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    n_query, n_doc = rel_mat.shape
    K = v.shape[0]
    unif_pi = np.ones((n_query, n_doc, K)) / n_doc
    expo_mat = (unif_pi * v.T).sum(2)
    click_mat: np.ndarray = rel_mat * expo_mat
    item_utils: np.ndarray = click_mat.sum(0) / n_query

    return item_utils


# ref: https://github.com/usaito/kdd2022-fair-ranking-nsw
def evaluate_pi(pi: np.ndarray, rel_mat: np.ndarray, v: np.ndarray) -> dict[str, float]:
    n_query, n_doc = rel_mat.shape
    expo_mat = (pi * v.T).sum(2)
    click_mat: np.ndarray = rel_mat * expo_mat
    user_util = click_mat.sum() / n_query
    item_utils: np.ndarray = click_mat.sum(0) / n_query
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
