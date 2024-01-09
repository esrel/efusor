""" eFusor utils """

__author__ = "Evgeny A. Stepanov"
__email__ = "stepanov.evgeny.a@gmail.com"
__status__ = "dev"
__version__ = "0.1.0"


import numpy as np


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


def softmax(vector: np.ndarray) -> np.ndarray:
    """
    numerically stable softmax with nan support
    :param vector: predictions scores (not probability)
    :type vector: np.ndarray
    :return: softmax
    :rtype: np.ndarray
    """
    return np.exp(vector - np.nanmax(vector))/np.nansum(np.exp(vector - np.nanmax(vector)))
