""" eFusor: scaling/normalization functions """

__author__ = "Evgeny A. Stepanov"
__email__ = "stepanov.evgeny.a@gmail.com"
__status__ = "dev"
__version__ = "0.1.0"


import numpy as np


def scale(vector: np.ndarray) -> np.ndarray:
    """
    min-max scale the vector in 0-1 range

    norm_vector = (vector - min(vector))/(max(vector) - min(vector))

    :param vector: vector to scale
    :type vector: np.ndarray
    :return: vector
    :rtype: np.ndarray
    """
    if np.isnan(vector).all():
        return vector

    if np.nanmin(vector) == np.nanmax(vector):
        return vector

    return (vector - np.nanmin(vector)) / (np.nanmax(vector) - np.nanmin(vector))
