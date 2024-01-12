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
from efusor.utils import softmax


def fuse(tensor: list | np.ndarray,
         method: str = "hard_voting",
         weights: list | np.ndarray = None,
         *,
         cutoff: float = None,
         scaled: bool = False,
         digits: int = None
         ) -> list:
    """
    fusion methods wrapper
    :param tensor: prediction scores as a tensor
    :type tensor: np.ndarray
    :param method: fusion method; defaults to "hard_voting"
    :type method: str, optional
    :param weights: predictor weights; defaults to None
    :type weights: np.ndarray, optional
    :param cutoff: prediction cut-off threshold; defaults to None
    :type cutoff: float, optional
    :param scaled: if to re-scale final scores (softmax); defaults to False
    :type scaled: bool, optional
    :param digits: rounding precision; defaults to None
    :type digits: int, optional
    :return: fused scores
    :rtype: np.ndarray
    """
    tensor = np.array(tensor) if isinstance(tensor, list) else tensor

    if cutoff:
        tensor[tensor < cutoff] = np.nan

    weights = np.array(weights) if isinstance(weights, list) else weights

    if method in {"hard_voting", "soft_voting", "majority_voting"}:
        result = vote(tensor, method=method, weights=weights)
    elif method in {"min", "max", "sum", "product", "median", "average", "mean"}:
        result = apply(tensor, method=method)
    elif method in {"borda"}:
        result = borda(tensor)
    elif method in {"priority"}:
        if weights is None:
            raise ValueError(f"method '{method}' requires weights")
        result = prioritize(tensor, weights)
    else:
        raise ValueError(f"unsupported fusion method: {method}")

    result = np.apply_along_axis(softmax, -1, result) if scaled else result
    result = np.round(result,  decimals=digits) if digits else result

    return result.tolist()
