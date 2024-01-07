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


def vectorize(labels: list[str], scores: dict | list) -> np.ndarray:
    """
    convert sequences of scores & labels into a vector (1d numpy array)
    :param labels: ordered list of labels/classes for vectorization
    :type labels: list[str]
    :param scores: predictions as {label: score}
    :type scores: dict | list
    :return: array
    :rtype: np.ndarray
    """
    return (batch(*[vectorize(labels, x) for x in scores]) if isinstance(scores, list) else
            np.array([scores.get(x, np.nan) for x in labels]))


def batch(*vector: np.ndarray) -> np.ndarray:
    """
    batch vectors into a matrix
    :param vector:
    :type vector: np.ndarray
    :return: matrix
    :rtype: np.ndarray
    """
    return np.stack(vector)


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
