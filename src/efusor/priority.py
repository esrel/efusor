""" eFusor: priority fusion """

__author__ = "Evgeny A. Stepanov"
__email__ = "stepanov.evgeny.a@gmail.com"
__status__ = "dev"
__version__ = "0.1.0"


import warnings

import numpy as np

from efusor.ranker import rank
from efusor.utils import batch


def ranked_slice(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    fuse numpy matrix of prediction scores w.r.t. priorities (weights): slice top ranked & max
    :param matrix: prediction scores as a matrix
    :type matrix: np.ndarray
    :param weights: predictor weights
    :type weights: np.ndarray
    :return: fused scores
    :rtype: np.ndarray
    """
    if matrix.ndim != 2:
        raise ValueError(f"unsupported input: array dim = {matrix.ndim}")

    ranks = rank(weights)
    index = [[i for i, y in enumerate(ranks.tolist()) if y == x] for x in set(ranks.tolist())]

    for indices in index:
        sliced = matrix[np.array(indices)]
        if not np.isnan(sliced).all():
            # suppress warnings for all-NaN columns
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                return np.nanmax(sliced, axis=-2)

    return np.full(matrix.shape[1], np.nan)


def prioritize(tensor: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    fuse numpy matrix of prediction scores w.r.t. priorities (weights): slice top ranked
    :param tensor: prediction scores as a tensor
    :type tensor: np.ndarray
    :param weights: predictor weights
    :type weights: np.ndarray
    :return: fused scores
    :rtype: np.ndarray
    """
    return (batch(*[ranked_slice(m, weights) for m in tensor]) if tensor.ndim > 2 else
            ranked_slice(tensor, weights))


def increment(tensor: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    fuse numpy tensor of scores w.r.t. priorities (weights): increment w.r.t. priority
    :param tensor: prediction scores as a tensor
    :type tensor: np.ndarray
    :param weights: predictor weights; defaults to None
    :type weights: np.ndarray, optional
    :return: fused scores
    :rtype: np.ndarray
    """
    addend = np.full(weights.shape, weights.size, dtype=int) - rank(weights)
    result = np.apply_along_axis(lambda m, v: (m.T + v).T, -2, tensor, addend)
    return np.nanmax(result, axis=-2)
