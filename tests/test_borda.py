""" test borda fusion """

import pytest

import numpy as np

from efusor.borda import borda, score, borda_score


def test_score_tensor(scores: list, borda_scores: list) -> None:
    """
    test borda's score on tensor input
    :param scores: scores
    :type scores: list
    :param borda_scores: borda scores
    :type borda_scores: list
    """
    assert borda_scores == score(np.array(scores)).tolist()


def test_score_matrix(scores: list, borda_scores: list) -> None:
    """
    test borda's score on matrix input
    :param scores: scores
    :type scores: list
    :param borda_scores: borda scores
    :type borda_scores: list
    """
    for i, matrix in enumerate(scores):
        assert score(np.array(matrix)).tolist() == borda_scores[i]


def test_score_vector(scores: list, borda_scores: list, borda_simple: list) -> None:
    """
    test borda's score on vector input
    :param scores: scores
    :type scores: list
    :param borda_scores: borda scores
    :type borda_scores: list
    :param borda_simple: simple borda scores (non-tournament)
    :type borda_simple: list
    """
    for i, matrix in enumerate(scores):
        for j, vector in enumerate(matrix):
            assert score(np.array(vector)).tolist() == borda_scores[i][j]
            assert borda_score(np.array(vector)).tolist() == borda_scores[i][j]
            assert borda_score(np.array(vector), tournament=False).tolist() == borda_simple[i][j]


def test_borda_tensor(scores: list, borda_result: list) -> None:
    """
    test borda on tensor
    :param scores: scores
    :type scores: list
    :param borda_result: results of borda voting
    :type borda_result: list
    """
    assert borda(np.array(scores)).tolist() == borda_result


def test_borda_matrix(scores: list, borda_result: list) -> None:
    """
    test borda on matrix
    :param scores: scores
    :type scores: list
    :param borda_result: results of borda voting
    :type borda_result: list
    """
    for i, matrix in enumerate(scores):
        assert borda(np.array(matrix)).tolist() == borda_result[i]


def test_borda_vector(scores: list) -> None:
    """
    test borda on vector
    :param scores: scores
    :type scores: list
    """
    for matrix in scores:
        for vector in matrix:
            with pytest.raises(np.AxisError):
                borda(np.array(vector))
