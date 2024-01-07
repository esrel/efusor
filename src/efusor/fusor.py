""" eFusor: fusion functions """

__author__ = "Evgeny A. Stepanov"
__email__ = "stepanov.evgeny.a@gmail.com"
__status__ = "dev"
__version__ = "0.1.0"


import numpy as np

from efusor.voter import vote
from efusor.basic import apply
from efusor.borda import borda
from efusor.priority import prioritize


def fuse(tensor: list | np.ndarray,
         method: str = "hard_voting",
         weights: list | np.ndarray = None
         ) -> list:
    """
    fusion methods wrapper
    :param tensor: prediction scores as a tensor
    :type tensor: np.ndarray
    :param method: fusion method; defaults to "hard_voting"
    :type method: str, optional
    :param weights: predictor weights; defaults to None
    :type weights: np.ndarray, optional
    :return: fused scores
    :rtype: np.ndarray
    """
    tensor = np.array(tensor) if isinstance(tensor, list) else tensor
    weights = np.array(weights) if isinstance(weights, list) else weights

    if method in {"hard_voting", "soft_voting", "majority_voting"}:
        result = vote(tensor, method=method, weights=weights)
    elif method in {"min", "max", "sum", "product", "median", "average", "mean"}:
        result = apply(tensor, method=method)
    elif method in {"borda"}:
        result = borda(tensor)
    elif method in {"priority"}:
        result = prioritize(tensor, weights)
    else:
        raise ValueError(f"unsupported fusion method: {method}")

    return result.tolist()
