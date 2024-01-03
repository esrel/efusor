""" eFusor: fusion functions """

__author__ = "Evgeny A. Stepanov"
__email__ = "stepanov.evgeny.a@gmail.com"
__status__ = "dev"
__version__ = "0.1.0"


import numpy as np

from efusor.voter import vote
from efusor.basic import apply
from efusor.borda import borda


def vectorize(scores: list[float]) -> np.ndarray:
    """
    convert sequences of scores & labels into a vector (1d numpy array)
    :param scores: sequence of prediction scores
    :type scores: list[float]
    :return: vector
    :rtype: np.ndarray
    """
    return np.array(scores)


def batch(*vector: np.ndarray) -> np.ndarray:
    """
    batch vectors into a matrix
    :param vector:
    :type vector: np.ndarray
    :return: matrix
    :rtype: np.ndarray
    """
    return np.stack(vector)


def fuse(tensor: np.ndarray,
         method: str = "hard_voting",
         weights: np.ndarray = None
         ) -> np.ndarray:
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
    if method in ["hard_voting", "soft_voting", "majority_voting"]:
        result = vote(tensor, method=method, weights=weights)
    elif method in ["min", "max", "sum", "product", "median", "average", "mean"]:
        result = apply(tensor, method=method)
    elif method in ["borda"]:
        result = borda(tensor)
    else:
        raise ValueError(f"unsupported fusion method: {method}")

    return result
