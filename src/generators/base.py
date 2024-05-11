from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from numpy.typing import NDArray


class BaseGenerator(ABC):
    @abstractmethod
    def generate_rel_mat(self) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        """Generate relevance matrices.

        Args:
            n_query (int): Number of queries.
            n_doc (int): Number of documents.

        Returns:
            tuple[NDArray[np.float_], NDArray[np.float_]]: True and observed relevance matrices.
        """
        raise NotImplementedError

    @staticmethod
    def exam_func(K: int, shape: Literal["inv", "exp"]="inv") -> NDArray[np.float_]:
        if shape == "inv":
            v = np.ones(K) / np.arange(1, K + 1)
        elif shape == "exp":
            v = 1.0 / np.exp(np.arange(K))
        else:
            raise ValueError(f"Invalid shape: {shape}, must be 'inv' or 'exp'")

        return v[:, np.newaxis]
