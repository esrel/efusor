""" eFusor: basic vector operations for fusion """

__author__ = "Evgeny A. Stepanov"
__email__ = "stepanov.evgeny.a@gmail.com"
__status__ = "dev"
__version__ = "0.1.0"


import numpy as np


def apply(tensor: np.ndarray, method: str = "max") -> np.ndarray:
    """
    apply basic classifier combination schemes

    Kittler, Hatef, Duin, and Matas (1998) "On Combining Classifiers".
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 20-3.

    n = # of classes      np.size(matrix, 1) | np.size(tensor, -1)
    r = # of classifiers  np.size(matrix, 0) | np.size(tensor, -2)

    Methods:
        # main
        - product: product rule
        - sum    : assumes that postriors are not far from priors
        - median : robust average assuming equal priors
        - min    : product is bound by min
        - max    : sum is approximated by max

        # added (use at your own risk):
        - mean   : simple average

    e.g. majority voting is a special case of 'hardened' sum

    :param tensor: scores tensor
    :type tensor: np.ndarray
    :param method: fusion method/operation; defaults to "max"
    :type method: str, optional
    :return: fused result
    :rtype: np.ndarray
    """

    if tensor.ndim == 1:
        return tensor

    labels = np.size(tensor, -1)  # number of classes
    models = np.size(tensor, -2)  # number of classifiers

    if labels == 0:
        return np.empty(shape=(0, 0))

    priors = 1 / labels  # assuming equal class priors

    match method:
        case "mean" | "average":
            result = np.nanmean(tensor, axis=-2)
        case "max":
            result = np.nanmax(tensor, axis=-2)
        case "min":
            result = np.nanmin(tensor, axis=-2)
        case "sum":
            result = np.nansum(tensor, axis=-2) + (1 - models) * priors
        case "product":
            result = np.nanprod(tensor, axis=-2) * priors
        case "median":
            result = np.nanmedian(tensor, axis=-2)
        case _:
            result = np.nanmax(tensor, axis=-2)

    result[result < 0] = 0

    return result
