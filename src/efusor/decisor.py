""" eFusor: decision functions """

__author__ = "Evgeny A. Stepanov"
__email__ = "stepanov.evgeny.a@gmail.com"
__status__ = "dev"
__version__ = "0.1.0"


import numpy as np


def rerank(tensor: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    rerank tensor w.r.t. vector
    :param tensor: tensor of prediction scores
    :type tensor: np.ndarray
    :param vector: vector of model weights
    :type vector: np.ndarray
    :return: re-ranked tensor
    :rtype: np.ndarray
    """
    return np.apply_along_axis(lambda m, v: (m.T * v).T, -2, tensor, vector)


def select(tensor: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    select row w.r.t. vector
    :param tensor: tensor of prediction scores
    :type tensor: np.ndarray
    :param vector: vector of model weights
    :type vector: np.ndarray
    :return: matrix
    :rtype: np.ndarray
    """
    return np.take(tensor, vector.argmax(), axis=-2)
