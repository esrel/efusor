""" test eFusor fuse """

import pytest

import numpy as np

from efusor.fusor import fuse, batch, vectorize


def test_batch_tensor(scores: list) -> None:
    """
    test batch
    :param scores: prediction scores
    :type scores: list
    """
    assert np.array_equal(batch(*[np.array(x) for x in scores]), np.array(scores), equal_nan=True)


def test_batch_matrix(scores: list) -> None:
    """
    test batch
    :param scores: prediction scores
    :type scores: list
    """
    for matrix in scores:
        assert np.array_equal(batch(*[np.array(x) for x in matrix]),
                              np.array(matrix), equal_nan=True)


def test_batch_vector(scores: list) -> None:
    """
    test batch
    :param scores: prediction scores
    :type scores: list
    """
    for matrix in scores:
        for vector in matrix:
            assert np.array_equal(batch(*[np.array(x) for x in vector]),
                                  np.array(vector), equal_nan=True)


def test_batch_scalar(scores: list) -> None:
    """
    test batch
    :param scores: prediction scores
    :type scores: list
    """
    for matrix in scores:
        for vector in matrix:
            for scalar in vector:
                with pytest.raises(TypeError):
                    batch(*[np.array(x) for x in scalar])


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


def test_vectorize_tensor(scores: list) -> None:
    """
    test vectorize
    :param scores: prediction scores
    :type scores: list
    """
    labels = ["A", "B", "C"]
    inputs = [[(dict(zip(labels, v)) if not np.isnan(v).all() else {}) for v in m] for m in scores]
    assert np.array_equal(vectorize(labels, inputs), np.array(scores), equal_nan=True)


def test_vectorize_matrix(scores: list) -> None:
    """
    test vectorize
    :param scores: prediction scores
    :type scores: list
    """
    labels = ["A", "B", "C"]
    inputs = [[(dict(zip(labels, v)) if not np.isnan(v).all() else {}) for v in m] for m in scores]

    for i, matrix in enumerate(inputs):
        assert np.array_equal(vectorize(labels, matrix), np.array(scores[i]), equal_nan=True)


def test_vectorize_vector(scores: list) -> None:
    """
    test vectorize
    :param scores: prediction scores
    :type scores: list
    """
    labels = ["A", "B", "C"]
    inputs = [[(dict(zip(labels, v)) if not np.isnan(v).all() else {}) for v in m] for m in scores]

    for i, matrix in enumerate(inputs):
        for j, vector in enumerate(matrix):
            assert np.array_equal(vectorize(labels, vector),
                                  np.array(scores[i][j]), equal_nan=True)
