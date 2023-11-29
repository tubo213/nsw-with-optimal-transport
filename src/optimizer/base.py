from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans


class BaseOptimizer(ABC):
    @abstractmethod
    def solve(self, rel_mat: np.ndarray, expo: np.ndarray) -> np.ndarray:
        """

        Args:
            rel_mat (np.ndarray): relevance matrix. (n_query, n_doc)
            expo (np.ndarray): exposure matrix. (n_rank, 1)

        Returns:
            np.ndarray: pi matrix. (n_query, n_doc, n_rank)
        """
        raise NotImplementedError


class BaseClusteredOptimizer(BaseOptimizer):
    def __init__(
        self,
        n_doc_cluster: Optional[int] = None,
        n_query_cluster: Optional[int] = None,
        random_state: int = 12345,
    ):
        # n_doc_cluster か n_query_clusterは必ず指定する
        assert (
            n_doc_cluster is not None or n_query_cluster is not None
        ), "n_doc_cluster or n_query_cluster must be specified."
        # 指定されている場合は, それぞれの値は1以上である必要がある
        assert n_doc_cluster is None or n_doc_cluster > 0, "n_doc_cluster must be positive."
        assert n_query_cluster is None or n_query_cluster > 0, "n_query_cluster must be positive."

        self.n_doc_cluster = n_doc_cluster
        self.n_query_cluster = n_query_cluster
        self.random_state = random_state

    def solve(self, rel_mat: np.ndarray, expo: np.ndarray) -> np.ndarray:
        """

        Args:
            rel_mat (np.ndarray): relevance matrix. (n_query, n_doc)
            expo (np.ndarray): exposure matrix. (n_rank, 1)

        Returns:
            np.ndarray: pi matrix. (n_query, n_doc, n_rank)
        """
        rel_mat, high, item_cluster_ids, query_cluster_ids = self._clustering(rel_mat)
        pi = self._solve(rel_mat, expo, high)
        pi = pi / high[None, :, None]

        return pi[query_cluster_ids][:, item_cluster_ids]

    def _clustering(
        self, rel_mat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Clustering relevance matrix.
        1. Cluster items
        2. Cluster queries

        Args:
            rel_mat (np.ndarray): relevance matrix. (n_query, n_doc)

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                (rel_mat, high, item_cluster_ids, query_cluster_ids)
        """
        # item clustering
        if self.n_doc_cluster is None:
            item_cluster_ids = np.arange(rel_mat.shape[1])
            high = np.ones(rel_mat.shape[1])  # n_doc_cluster
        else:
            kmeans = KMeans(
                n_clusters=self.n_doc_cluster, random_state=self.random_state, n_init="auto"
            )
            kmeans.fit(rel_mat.T)
            item_cluster_ids = kmeans.predict(rel_mat.T)
            rel_mat = kmeans.cluster_centers_.T
            _, high = np.unique(item_cluster_ids, return_counts=True)

        # query clustering
        if self.n_query_cluster is None:
            query_cluster_ids = np.arange(rel_mat.shape[0])
        else:
            kmeans = KMeans(
                n_clusters=self.n_query_cluster, random_state=self.random_state, n_init="auto"
            )
            kmeans.fit(rel_mat)
            query_cluster_ids = kmeans.predict(rel_mat)
            rel_mat = kmeans.cluster_centers_

        return rel_mat, high, item_cluster_ids, query_cluster_ids

    @abstractmethod
    def _solve(self, rel_mat: np.ndarray, expo: np.ndarray, high: np.ndarray) -> np.ndarray:
        """

        Args:
            rel_mat (np.ndarray): relevance matrix. (n_query_cluster, n_doc_cluster)
            expo (np.ndarray): exposure matrix. (n_rank, 1)
            high (np.ndarray): number of items in each cluster. (n_doc_cluster)

        Returns:
            np.ndarray: pi matrix. (n_query_cluster, n_doc_cluster, n_rank)
        """
        raise NotImplementedError
