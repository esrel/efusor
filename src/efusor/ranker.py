""" eFusor: ranking functions """

__author__ = "Evgeny A. Stepanov"
__email__ = "stepanov.evgeny.a@gmail.com"
__status__ = "dev"
__version__ = "0.1.0"


import numpy as np


def rank(tensor: np.ndarray) -> np.ndarray:
    """
    rank tensor (or matrix or vector) values
    :param tensor:
    :return:
    :type tensor: nd.array
    :rtype: np.ndarray
    """
    return np.apply_along_axis(rank_vector, -1, tensor)


def rank_vector(vector: np.ndarray) -> np.ndarray:
    """
    create array of ranks w.r.t. array of values such that the highest value has the lowest rank
    takes care of duplicate values, assigning them the same rank
    :param vector: vector
    :type vector: np.ndarray
    :return: vector
    :rtype: np.ndarray
    """
    vector = np.nan_to_num(vector)
    values, indices = np.unique(vector, return_inverse=True)
    ranked = np.argsort(-values)
    result = np.take(ranked, indices)
    return result
