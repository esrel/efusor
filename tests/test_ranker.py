""" eFusor ranker tests """

import pytest

import numpy as np

from efusor.ranker import rank, rank_vector


def test_rank_tensor(scores: list, ranks: list) -> None:
    """
    test ranking
    :param scores: scores to rank
    :type scores: list
    :param ranks: reference ranks
    :type ranks: list
    """
    assert rank(np.array(scores)).tolist() == ranks


def test_rank_matrix(scores: list, ranks: list) -> None:
    """
    test ranking
    :param scores: scores to rank
    :type scores: list
    :param ranks: reference ranks
    :type ranks: list
    """
    for i, matrix in enumerate(scores):
        assert rank(np.array(matrix)).tolist() == ranks[i]


def test_rank_vector(scores: list, ranks: list) -> None:
    """
    test ranking
    :param scores: scores to rank
    :type scores: list
    :param ranks: reference ranks
    :type ranks: list
    """
    for i, matrix in enumerate(scores):
        for j, vector in enumerate(matrix):
            assert rank(np.array(vector)).tolist() == ranks[i][j]
            assert rank_vector(np.array(vector)).tolist() == ranks[i][j]


def test_rank_scalar(scores: list) -> None:
    """
    test ranking
    :param scores: scores to rank
    :type scores: list
    """
    for matrix in scores:
        for vector in matrix:
            for scalar in vector:
                with pytest.raises(np.AxisError):
                    rank(np.array(scalar))


@pytest.mark.parametrize("vector, ranked", [
    ([0.0, 0.1, 0.2, 0.7], [3, 2, 1, 0]),
    ([0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0]),
    ([0.0, 0.1, 0.1, 0.8], [2, 1, 1, 0]),
])
def test_rank_vector_multi(vector: list, ranked: list) -> None:
    """
    test different ranking input
    :param vector: vector to rank
    :type vector: list
    :param ranked: vector ranks
    :type ranked: list
    """
    assert rank_vector(np.array(vector)).tolist() == ranked
