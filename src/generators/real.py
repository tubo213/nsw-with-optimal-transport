import pickle
from functools import wraps
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from ._pyxclib_utils import read_data  # type: ignore
from ._registry import register_generator
from .base import BaseGenerator

__all__ = ["RealGenerator", "real"]


def cache_function_output() -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator that caches the output of a function based on its arguments.

    The decorator generates a unique key based on the function name and its arguments.
    It checks if the cached data exists for the key, and if so, returns the cached result.
    If the data is not cached, it executes the function and caches the result for future use.

    Returns:
        A decorator function that can be applied to other functions.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generate a unique key based on the function name and arguments
            args_repr = (
                "_".join(map(str, args)) + "_" + "_".join(f"{k}={v}" for k, v in kwargs.items())
            )
            hash_key = sha256(args_repr.encode()).hexdigest()
            cache_dir = Path.home() / ".cache" / "nsw-with-optimal-transport" / str(func.__name__)
            cache_dir.mkdir(parents=True, exist_ok=True)
            file_path = cache_dir / f"{hash_key}.pkl"

            # If cached data exists, load and return it
            if file_path.exists():
                logger.info(f"Loading cached data for {func.__name__} from {file_path}")
                with file_path.open("rb") as file:
                    return pickle.load(file)

            # Execute the function and cache the result
            result = func(*args, **kwargs)
            with file_path.open("wb") as file:
                logger.info(f"Caching data for {func.__name__} to {file_path}")
                pickle.dump(result, file)
            return result

        return wrapper

    return decorator


class RealGenerator(BaseGenerator):
    def __init__(
        self,
        dataset: Literal["d", "w"] = "d",
        path: str = "./data",
        n_doc: int = 100,
        lam: float = 1.0,
        test_size: float = 0.1,
        eps_plus: float = 1.0,
        eps_minus: float = 0.0,
        random_state: int = 12345,
    ) -> None:
        self.dataset = dataset
        self.path = path
        self.n_doc = n_doc
        self.lam = lam
        self.test_size = test_size
        self.eps_plus = eps_plus
        self.eps_minus = eps_minus
        self.random_state = random_state

    def generate_rel_mat(
        self,
    ) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        rel_mat_true, rel_mat_obs, _ = self.preprocess_data(
            self.dataset,
            self.path,
            self.n_doc,
            self.lam,
            self.test_size,
            self.eps_plus,
            self.eps_minus,
            self.random_state,
        )
        return rel_mat_true, rel_mat_obs

    @staticmethod
    @cache_function_output()
    def preprocess_data(
        dataset: Literal["d", "w"] = "d",
        path: str = "./data",
        n_doc: int = 100,
        lam: float = 1.0,
        test_size: float = 0.1,
        eps_plus: float = 1.0,
        eps_minus: float = 0.0,
        random_state: int = 12345,
    ) -> tuple[NDArray[np.float_], NDArray[np.float_], dict[str, Any]]:
        """
        Preprocesses the data for training and testing.

        Args:
            dataset (str): The dataset to use. Possible values are "d" for "delicious" and "w" for "wiki".
            path (Path, optional): The path to the data directory. Defaults to Path("./data").
            n_doc (int, optional): The number of documents to sample. Defaults to 100.
            lam (float, optional): The lambda value for sampling labels. Defaults to 1.0.
            test_size (float, optional): The proportion of the data to use for testing. Defaults to 0.1.
            eps_plus (float, optional): The epsilon plus value for adding noise to the train data. Defaults to 1.0.
            eps_minus (float, optional): The epsilon minus value for adding noise to the train data. Defaults to 0.0.
            random_state (int, optional): The random state for reproducibility. Defaults to 12345.

        Returns:
            tuple[NDArray[np.float_], NDArray[np.float_], dict[str, Any]]: A tuple containing the true relevance matrix for the test data, the predicted relevance matrix for the test data, and additional information about the data preprocessing.

        References:
            - This function is adapted from `preprocess_data` at [https://github.com/usaito/kdd2022-fair-ranking-nsw/blob/main/src/real/func.py) by yuta-saito.
            - MIT License, Copyright (c) 2022 yuta-saito. See the original repository for full license information.
        """
        info = dict()
        random_ = check_random_state(random_state)
        path = Path(path)

        # Read file with features and labels
        if dataset == "d":
            dataset_ = "delicious"
        elif dataset == "w":
            dataset_ = "wiki"
        features, tabels, num_samples, num_features, num_labels = read_data(
            path / f"{dataset_}.txt"
        )
        info["num_all_data"] = num_samples
        info["num_features"] = num_features
        info["num_labels"] = num_labels
        info["n_doc"] = n_doc

        # BoW Feature
        X: NDArray[np.float_] = features.toarray()
        # Multilabel Table
        T = tabels.toarray()

        # sample labels via eps-greedy
        n_points_per_label = T.sum(0)
        top_labels = rankdata(-n_points_per_label, method="ordinal") <= n_doc
        sampling_probs = lam * top_labels + (1 - lam) * n_doc / num_labels
        sampling_probs /= sampling_probs.sum()
        sampled_labels = random_.choice(
            np.arange(num_labels), size=n_doc, p=sampling_probs, replace=False
        )
        T = T[:, sampled_labels]
        info["lam"] = lam

        # minimum num of relevant labels per data
        if dataset == "d":
            n_rel_labels = 10
        elif dataset == "w":
            n_rel_labels = 5
        n_rel_labels = np.maximum(n_rel_labels * lam, 1)
        n_labels_per_point = T.sum(1)
        T = T[n_labels_per_point >= n_rel_labels, :]
        X = X[n_labels_per_point >= n_rel_labels, :]

        # train-test split
        X_tr: NDArray[np.float_]
        X_te: NDArray[np.float_]
        rel_mat_tr: NDArray[np.float_]
        rel_mat_te: NDArray[np.float_]
        X_tr, X_te, rel_mat_tr, rel_mat_te = train_test_split(
            X,
            T,
            test_size=test_size,
            random_state=random_state,
        )
        rel_mat_obs = rel_mat_tr[:, (rel_mat_tr.sum(0) != 0) & (rel_mat_te.sum(0) != 0)]
        rel_mat_te = rel_mat_te[:, (rel_mat_tr.sum(0) != 0) & (rel_mat_te.sum(0) != 0)]
        info["num_train"] = X_tr.shape[0]
        info["num_test"] = X_te.shape[0]

        # add noise to train data
        rel_mat_tr_prob = eps_plus * rel_mat_obs + eps_minus * (1 - rel_mat_obs)
        rel_mat_tr_obs = random_.binomial(n=1, p=rel_mat_tr_prob)

        # relevance prediction
        n_components = np.minimum(200, rel_mat_obs.shape[0])
        classifier = LogisticRegression(
            C=100,
            max_iter=1000,
            random_state=12345,
        )
        pipe = Pipeline(
            [
                ("pca", PCA(n_components=n_components)),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MultiOutputClassifier(
                        classifier,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        pipe.fit(X_tr, rel_mat_tr_obs)
        rel_mat_te_pred = np.concatenate(
            [_[:, 1][:, np.newaxis] for _ in pipe.predict_proba(X_te)], 1
        )
        info["explained_variance_ratio"] = pipe["pca"].explained_variance_ratio_.sum()
        info["log_loss"] = log_loss(rel_mat_te.flatten(), rel_mat_te_pred.flatten())
        info["auc"] = roc_auc_score(rel_mat_te.flatten(), rel_mat_te_pred.flatten())

        return rel_mat_te, rel_mat_te_pred, info


@register_generator
def real(**kwargs: Any) -> RealGenerator:
    return RealGenerator(**kwargs)
