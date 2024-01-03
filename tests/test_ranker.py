""" eFusor ranker tests """

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
