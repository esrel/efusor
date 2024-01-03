""" eFusor: voting functions """

__author__ = "Evgeny A. Stepanov"
__email__ = "stepanov.evgeny.a@gmail.com"
__status__ = "dev"
__version__ = "0.1.0"


import numpy as np


def harden(vector: np.ndarray) -> np.ndarray:
    """
    "harden" vector: max is 1, rest is 0
    :param vector: vector to harden
    :type vector: np.ndarray
    :return: hardened vector
    :rtype: np.ndarray
    """
    vector = np.nan_to_num(vector)
    vector = np.where((vector == np.max(vector)) & (vector != 0), 1, 0)
    return vector


def nanaverage(array: np.ndarray,
               axis: int = None,
               weights: np.ndarray = None
               ) -> int | float | np.ndarray:
    """
    average vector w.r.t. weights, if provided

    computes sum(w_i * x_i) / sum(w), if weights

    :param array: array to average
    :type array: np.ndarray
    :param axis: axis to average on; defaults to None
    :type axis: int, optional
    :param weights: weight for averaging; defaults to None
    :type weights: np.ndarray, optional
    :return: averaged array
    :rtype: int | float | np.ndarray
    """
    msk_array = np.ma.MaskedArray(array, mask=np.isnan(array))
    avg_array = np.ma.average(msk_array, axis=axis, weights=weights)
    return avg_array.data if isinstance(array, np.ndarray) else avg_array


def vote(tensor: np.ndarray,
         method: str = "hard_voting",
         weights: np.ndarray = None
         ) -> np.ndarray:
    """
    Voting Fusion

    - scikit-learn VotingClassifier
    - http://rasbt.github.io/mlxtend/user_guide/classifier/EnsembleVoteClassifier/

    Implements "hard" and "soft" voting. In hard voting,
    we predict the final class label as the class label that has
    been predicted most frequently by the classification models.
    In soft voting, we predict the class labels by averaging the class-probabilities
    (only recommended if the classifiers are well-calibrated).

    Methods:
        - hard_voting: same as majority voting
        - soft_voting: majority voting with classifier weights

    :param tensor: prediction scores as a tensor
    :type tensor: np.ndarray
    :param method: fusion method; defaults to "hard_voting"
    :type method: str, optional
    :param weights: predictor weights; defaults to None
    type weights: np.ndarray, optional
    :return: fused scores
    :rtype: np.ndarray
    """
    match method:
        case "hard_voting" | "majority_voting":
            bin_array = np.apply_along_axis(harden, -1, tensor)
            return (nanaverage(bin_array, weights=weights, axis=-2) if weights is not None else
                    np.sum(bin_array, axis=-2))
        case "soft_voting" | "average":
            return (nanaverage(tensor, weights=weights, axis=-2) if weights is not None else
                    np.nanmean(tensor, axis=-2))
        case _:
            return np.sum(np.apply_along_axis(harden, -1, tensor), axis=-2)
