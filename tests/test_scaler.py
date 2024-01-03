""" eFusor scaler tests """

import numpy as np

from efusor.scaler import scale


def test_scale(scores: list, norms: list) -> None:
    """
    test scale
    :param scores: prediction scores
    :type scores: list
    :param norms: scaled scores
    :type norms: list
    """
    for i, matrix in enumerate(scores):
        for j, vector in enumerate(matrix):
            if np.isnan(vector).all():
                assert np.isnan(scale(np.array(vector))).all()
            else:
                assert scale(np.array(vector)).round(2).tolist() == norms[i][j]
