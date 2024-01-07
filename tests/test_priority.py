""" test priority fusion """

import pytest

import numpy as np

from efusor.priority import prioritize, ranked_slice, increment


def test_ranked_slice_matrix(scores: list, weights: list, prioritized: list) -> None:
    """
    test ranked_slice (priority fusion on a matrix)
    :param scores: tensor prediction scores
    :type scores: list
    :param weights: vector of model weights
    :type weights: list
    :param prioritized: prioritized tensor slice
    :type prioritized: list
    """
    for i, matrix in enumerate(scores):
        assert np.array_equal(ranked_slice(np.array(matrix), np.array(weights)),
                              np.array(prioritized[i]))


def test_ranked_slice_tensor(scores: list, weights: list) -> None:
    """
    test ranked_slice (priority fusion on a matrix)
    :param scores: tensor prediction scores
    :type scores: list
    :param weights: vector of model weights
    :type weights: list
    """
    with pytest.raises(ValueError):
        ranked_slice(np.array(scores), np.array(weights))


def test_ranked_slice_vector(scores: list, weights: list) -> None:
    """
    test ranked_slice (priority fusion on a matrix)
    :param scores: tensor prediction scores
    :type scores: list
    :param weights: vector of model weights
    :type weights: list
    """
    for matrix in scores:
        for vector in matrix:
            with pytest.raises(ValueError):
                ranked_slice(np.array(vector), np.array(weights))


def test_ranked_slice_scalar(scores: list, weights: list) -> None:
    """
    test ranked_slice (priority fusion on a matrix)
    :param scores: tensor prediction scores
    :type scores: list
    :param weights: vector of model weights
    :type weights: list
    """
    for matrix in scores:
        for vector in matrix:
            for scalar in vector:
                with pytest.raises(ValueError):
                    ranked_slice(np.array(scalar), np.array(weights))


def test_prioritize_tensor(scores: list, weights: list, prioritized: list) -> None:
    """
    test priority fusion
    :param scores: tensor prediction scores
    :type scores: list
    :param weights: vector of model weights
    :type weights: list
    :param prioritized: prioritized tensor
    :type prioritized: list
    """
    assert np.array_equal(prioritize(np.array(scores), np.array(weights)), np.array(prioritized))


def test_prioritize_matrix(scores: list, weights: list, prioritized: list) -> None:
    """
    test priority fusion
    :param scores: tensor prediction scores
    :type scores: list
    :param weights: vector of model weights
    :type weights: list
    :param prioritized: prioritized tensor
    :type prioritized: list
    """
    for i, matrix in enumerate(scores):
        assert np.array_equal(prioritize(np.array(matrix), np.array(weights)),
                              np.array(prioritized[i]))


def test_prioritize_vector(scores: list, weights: list) -> None:
    """
    test priority fusion
    :param scores: tensor prediction scores
    :type scores: list
    :param weights: vector of model weights
    :type weights: list
    """
    for matrix in scores:
        for vector in matrix:
            with pytest.raises(ValueError):
                prioritize(np.array(vector), np.array(weights))


def test_prioritize_scalar(scores: list, weights: list) -> None:
    """
    test priority fusion
    :param scores: tensor prediction scores
    :type scores: list
    :param weights: vector of model weights
    :type weights: list
    """
    for matrix in scores:
        for vector in matrix:
            for scalar in vector:
                with pytest.raises(ValueError):
                    prioritize(np.array(scalar), np.array(weights))


@pytest.mark.parametrize("weights, results", [
    ([0.25, 0.50, 0.75, 0.90, 1.00], [np.nan, np.nan, 1.0]),
    ([0.25, 1.00, 0.75, 0.90, 1.00], [0.7, 0.2, 0.1]),
    ([0.25, 0.25, 0.25, 0.25, 0.25], [0.7, 0.3, 1.0])
])
def test_ranked_slice_multi(weights: list, results: list) -> None:
    """
    test different priority slicing
    :param weights: predictor weights
    :type weights: list
    :param results: fused vector
    :type results: list
    """
    matrix = [[0.2, 0.3, 0.5],
              [0.7, 0.2, 0.1],
              [0.0, 0.0, 0.0],
              [np.nan, np.nan, 1.0],
              [np.nan, np.nan, np.nan]]

    assert np.array_equal(ranked_slice(np.array(matrix), np.array(weights)),
                          np.array(results), equal_nan=True)


def test_increment_tensor(scores: list, weights: list, incremented: list) -> None:
    """
    test priority fusion: increment
    :param scores: tensor prediction scores
    :type scores: list
    :param weights: vector of model weights
    :type weights: list
    :param incremented: incremented & fused tensor
    :type incremented: list
    """
    assert np.array_equal(increment(np.array(scores), np.array(weights)),
                          np.array(incremented), equal_nan=True)


def test_increment_matrix(scores: list, weights: list, incremented: list) -> None:
    """
    test priority fusion: increment
    :param scores: tensor prediction scores
    :type scores: list
    :param weights: vector of model weights
    :type weights: list
    :param incremented: incremented & fused tensor
    :type incremented: list
    """
    for i, matrix in enumerate(scores):
        assert np.array_equal(increment(np.array(matrix), np.array(weights)),
                              np.array(incremented[i]), equal_nan=True)


def test_incremented_vector(scores: list, weights: list) -> None:
    """
    test priority fusion: increment
    :param scores: tensor prediction scores
    :type scores: list
    :param weights: vector of model weights
    :type weights: list
    """
    for matrix in scores:
        for vector in matrix:
            with pytest.raises(ValueError):
                increment(np.array(vector), np.array(weights))


def test_incremented_scalar(scores: list, weights: list) -> None:
    """
    test priority fusion: increment
    :param scores: tensor prediction scores
    :type scores: list
    :param weights: vector of model weights
    :type weights: list
    """
    for matrix in scores:
        for vector in matrix:
            for scalar in vector:
                with pytest.raises(ValueError):
                    increment(np.array(scalar), np.array(weights))
