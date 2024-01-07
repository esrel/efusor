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
