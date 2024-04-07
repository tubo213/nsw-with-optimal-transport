from src.optimizer.base import BaseOptimizer
from src.optimizer.greedy import GreedyOptimizer
from src.optimizer.nsw import ClusteredNSWOptimizer, NSWOptimizer
from src.optimizer.ot_osw import ClusteredOTNSWOptimizer, OTNSWOptimizer

VALID_OPTIMIZERS = ["greedy", "nsw", "clustered_nsw", "ot_nsw", "clustered_ot_nsw"]


def get_optimizer(name: str, params: dict) -> BaseOptimizer:
    """Get optimizer instance.

    Args:
        name (str): name of optimizer.
        params (dict): parameters of optimizer.

    Returns:
        BaseOptimizer: optimizer instance.
    """
    if name == "greedy":
        return GreedyOptimizer()
    elif name == "nsw":
        return NSWOptimizer(**params)
    elif name == "clustered_nsw":
        return ClusteredNSWOptimizer(**params)
    elif name == "ot_nsw":
        return OTNSWOptimizer(**params)
    elif name == "clustered_ot_nsw":
        return ClusteredOTNSWOptimizer(**params)
    else:
        raise ValueError(f"Invalid optimizer name: {name}, valid names are {VALID_OPTIMIZERS}")
