import numpy as np
from sklearn.utils import check_random_state

# ref: https://github.com/usaito/kdd2022-fair-ranking-nsw
def synthesize_rel_mat(
    n_query: int,
    n_doc: int,
    lam: float = 0.0,
    flip_ratio: float = 0.3,
    noise: float = 0.0,
    random_state: int = 12345,
) -> tuple[np.ndarray, np.ndarray]:
    random_ = check_random_state(random_state)

    # generate true relevance matrix
    rel_mat_unif = random_.uniform(0.0, 1.0, size=(n_query, n_doc))
    rel_mat_pop_doc = np.ones_like(rel_mat_unif)
    rel_mat_pop_doc -= (np.arange(n_doc) / n_doc)[np.newaxis, :]
    rel_mat_pop_query = np.ones_like(rel_mat_unif)
    rel_mat_pop_query -= (np.arange(n_query) / n_query)[:, np.newaxis]
    rel_mat_pop = rel_mat_pop_doc * rel_mat_pop_query
    rel_mat_pop *= rel_mat_unif.sum() / rel_mat_pop.sum()
    rel_mat_true = (1.0 - lam) * rel_mat_unif
    rel_mat_true += lam * rel_mat_pop

    # flipping
    flip_docs = random_.choice(np.arange(n_doc), size=np.int32(flip_ratio * n_doc), replace=False)
    flip_docs.sort()
    rel_mat_true[:, flip_docs] = rel_mat_true[::-1, flip_docs]

    # gen noisy relevance matrix
    if noise > 0.0:
        rel_mat_obs = np.copy(rel_mat_true)
        rel_mat_obs += random_.uniform(-noise, noise, size=(n_query, n_doc))
        rel_mat_obs = np.maximum(rel_mat_obs, 0.001)
    else:
        rel_mat_obs = rel_mat_true

    return rel_mat_true, rel_mat_obs


def exam_func(K: int, shape: str = "inv") -> np.ndarray:
    assert shape in ["inv", "exp", "log"]
    if shape == "inv":
        v = np.ones(K) / np.arange(1, K + 1)
    elif shape == "exp":
        v = 1.0 / np.exp(np.arange(K))

    return v[:, np.newaxis]
