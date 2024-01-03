""" Borda Count Decision Fusion """

__author__ = "Evgeny A. Stepanov"
__email__ = "stepanov.evgeny.a@gmail.com"
__status__ = "dev"
__version__ = "0.1.0"


import numpy as np


def borda(tensor: np.ndarray) -> np.ndarray:
    """
    apply borda count to a tensor (or a matrix)
    :param tensor: tensor or matrix
    :type tensor: np.ndarray
    :return: matrix
    :rtype: np.ndarray
    """
    return np.sum(score(tensor), axis=-2)


def score(tensor: np.ndarray) -> np.ndarray:
    """
    convert tensor (or matrix or vector) values to scores based on rank
    :param tensor:
    :type tensor: np.ndarray
    :return:
    :rtype: np.ndarray
    """
    return np.apply_along_axis(borda_score, -1, tensor)


def borda_score(vector: np.ndarray, tournament: bool = True) -> np.ndarray:
    """
    create array of scores w.r.t. array of values such that the highest value has the highest score
    takes care of duplicate values, assigning them the same score (from rank, starts from 1)

    computes: (score = n - rank - 1), where n is number of classes (length of a vector)

    :param vector:
    :param tournament:
    :return:
    :type vector: np.ndarray
    :type tournament: bool
    :rtype: np.ndarray
    """
    vector = np.nan_to_num(vector)
    values, indices, counts = np.unique(vector, return_counts=True, return_inverse=True)
    scores = np.argsort(values).astype(float)

    if scores.size != vector.size:
        if tournament:
            # if ranked (i.e. value > 0)
            # 1.0 point for each candidate preferred over
            # 0.5 point for each candidate tied with
            scores[(counts == 1) & (scores > 0)] += (vector.size - scores.size)
            points = (counts - 1) * 0.5
            points[values == 0] = 0
            scores += points
        else:
            scores[values > 0] += (vector.size - scores.size)

    result = np.take(scores, indices)
    return result
