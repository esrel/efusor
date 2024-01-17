""" test eFusor fuse """

import pytest

import numpy as np

from efusor.fusor import fuse


def test_fuse_tensor(scores: list, weights: list, weighted_hard_votes: list) -> None:
    """
    test fuse wrapper
    :param scores: prediction scores
    :type scores: list
    :param weights: predictor weights
    :type weights: list
    :param weighted_hard_votes: voting decision
    :type weighted_hard_votes: list
    """
    result = fuse(np.array(scores), method="hard_voting", weights=np.array(weights))
    assert result == weighted_hard_votes


def test_fuse_matrix(scores: list, weights: list, weighted_hard_votes: list) -> None:
    """
    test fuse wrapper
    :param scores: prediction scores
    :type scores: list
    :param weights: predictor weights
    :type weights: list
    :param weighted_hard_votes: voting decision
    :type weighted_hard_votes: list
    """
    for i, matrix in enumerate(scores):
        result = fuse(np.array(matrix), method="hard_voting", weights=np.array(weights))
        assert result == weighted_hard_votes[i]


def test_fuse_vector(scores: list, weights: list) -> None:
    """
    test fuse wrapper
    :param scores: prediction scores
    :type scores: list
    :param weights: predictor weights
    :type weights: list
    """
    for matrix in scores:
        for vector in matrix:
            with pytest.raises(IndexError):
                fuse(np.array(vector), method="hard_voting", weights=np.array(weights))


def test_fuse_scalar(scores: list, weights: list) -> None:
    """
    test fuse wrapper
    :param scores: prediction scores
    :type scores: list
    :param weights: predictor weights
    :type weights: list
    """
    for matrix in scores:
        for vector in matrix:
            for scalar in vector:
                with pytest.raises(IndexError):
                    fuse(np.array(scalar), method="hard_voting", weights=np.array(weights))


def test_fusor_cutoff(scores: list) -> None:
    """
    test cutoff
    :param scores: prediction scores
    :type scores: list
    """
    cutoff_max = [[0.7, 0.3, 0.5],
                  [0.4, 0.4, 0.6],
                  [0.4, 0.7, np.nan],
                  [0.3, 0.3, 1.0],
                  [np.nan, np.nan, np.nan]]

    result = fuse(np.array(scores), method="max", cutoff=0.1)
    assert np.array_equal(np.array(result), np.array(cutoff_max), equal_nan=True)


def test_fusor_scaled(scores: list) -> None:
    """
    test softmax
    :param scores: prediction scores
    :type scores: list
    """
    scaled_max = [[0.4, 0.27, 0.33],
                  [0.31, 0.31, 0.38],
                  [0.33, 0.45, 0.22],
                  [0.25, 0.25, 0.5],
                  [0.33, 0.33, 0.33]]

    cutoff_max = [[0.4, 0.27, 0.33],
                  [0.31, 0.31, 0.38],
                  [0.43, 0.57, np.nan],
                  [0.25, 0.25, 0.5],
                  [np.nan, np.nan, np.nan]]

    result = fuse(np.array(scores), method="max", scaled=True, digits=2)
    assert np.array_equal(np.array(result), np.array(scaled_max), equal_nan=True)

    result = fuse(np.array(scores), method="max", cutoff=0.1, scaled=True, digits=2)
    assert np.array_equal(np.array(result), np.array(cutoff_max), equal_nan=True)
