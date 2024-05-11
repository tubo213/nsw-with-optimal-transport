from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.utils import check_random_state

from ._registry import register_generator
from .base import BaseGenerator

__all__ = ["SyntheticGenerator"]


class SyntheticGenerator(BaseGenerator):
    def __init__(
        self,
        n_query: int,
        n_doc: int,
        lam: float = 0.0,
        flip_ratio: float = 0.3,
        noise: float = 0.0,
        random_state: int = 12345,
    ) -> None:
        self.n_query = n_query
        self.n_doc = n_doc
        self.lam = lam
        self.flip_ratio = flip_ratio
        self.noise = noise
        self.random_state = random_state

    def generate_rel_mat(self) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        return self.synthesize_rel_mat(
            self.n_query,
            self.n_doc,
            self.lam,
            self.flip_ratio,
            self.noise,
            self.random_state,
        )

    @staticmethod
    def synthesize_rel_mat(
        n_query: int,
        n_doc: int,
        lam: float = 0.0,
        flip_ratio: float = 0.3,
        noise: float = 0.0,
        random_state: int = 12345,
    ) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        """
        Synthesizes a relevance matrix for query-document pairs.

        This function generates a synthetic relevance matrix for a given number of queries and documents.
        The relevance matrix represents the relevance scores between each query and document pair.
        The generated matrix can be used for evaluating ranking algorithms.

        Args:
            n_query (int): The number of queries.
            n_doc (int): The number of documents.
            lam (float, optional): The weight parameter for combining the true relevance matrix and the popularity-based matrix. Defaults to 0.0.
            flip_ratio (float, optional): The ratio of documents to flip the relevance scores. Defaults to 0.3.
            noise (float, optional): The amount of noise to add to the relevance scores. Defaults to 0.0.
            random_state (int, optional): The random seed for reproducibility. Defaults to 12345.

        Returns:
            tuple[NDArray[np.float_], NDArray[np.float_]]: A tuple containing the true relevance matrix and the observed relevance matrix.
                The true relevance matrix represents the ground truth relevance scores between each query and document pair.
                The observed relevance matrix represents the noisy version of the true relevance matrix, with optional flipping and noise.

        References:
            - This function is adapted from `synthesize_rel_mat` at [https://github.com/usaito/kdd2022-fair-ranking-nsw/blob/main/src/synthetic/func.py) by yuta-saito.
            - MIT License, Copyright (c) 2022 yuta-saito. See the original repository for full license information.
        """
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
        flip_docs = random_.choice(
            np.arange(n_doc), size=np.int32(flip_ratio * n_doc), replace=False
        )
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


@register_generator
def synthetic(**kwargs: Any) -> SyntheticGenerator:
    return SyntheticGenerator(**kwargs)
