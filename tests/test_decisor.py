""" eFusor decisor tests """

import pytest

import numpy as np

from efusor.decisor import rerank, select


def test_rerank_tensor(scores: list, weights: list, re_ranked: list) -> None:
    """
    test rerank on tensor input
    :param scores: tensor prediction scores
    :type scores: list
    :param weights: vector of model weights
    :type weights: list
    :param re_ranked: re-ranked tensor
    :type re_ranked: list
    """
    assert np.array_equal(rerank(np.array(scores), np.array(weights)).round(3),
                          np.array(re_ranked), equal_nan=True)


def test_rerank_matrix(scores: list, weights: list, re_ranked: list) -> None:
    """
    test rerank on matrix input
    :param scores: tensor prediction scores
    :type scores: list
    :param weights: vector of model weights
    :type weights: list
    :param re_ranked: re-ranked tensor
    :type re_ranked: list
    """
    for i, matrix in enumerate(scores):
        assert np.array_equal(rerank(np.array(matrix), np.array(weights)).round(3),
                              np.array(re_ranked[i]), equal_nan=True)


def test_rerank_vector(scores: list, weights: list) -> None:
    """
    test rerank on vector input
    :param scores: tensor prediction scores
    :type scores: list
    :param weights: vector of model weights
    :type weights: list
    """
    for matrix in scores:
        for vector in matrix:
            with pytest.raises(np.AxisError):
                rerank(np.array(vector), np.array(weights))


def test_rerank_scalar(scores: list, weights: list) -> None:
    """
    test rerank on scalar input
    :param scores: tensor prediction scores
    :type scores: list
    :param weights: vector of model weights
    :type weights: list
    """
    for matrix in scores:
        for vector in matrix:
            for scalar in vector:
                with pytest.raises(np.AxisError):
                    rerank(np.array(scalar), np.array(weights))


def test_select_tensor(scores: list, weights: list, selected: list) -> None:
    """
    test select on tensor input
    :param scores: tensor prediction scores
    :type scores: list
    :param weights: vector of model weights
    :type weights: list
    :param selected: selected tensor
    :type selected: list
    """
    assert np.array_equal(select(np.array(scores), np.array(weights)),
                          np.array(selected), equal_nan=True)


def test_select_matrix(scores: list, weights: list, selected: list) -> None:
    """
    test select on matrix input
    :param scores: tensor prediction scores
    :type scores: list
    :param weights: vector of model weights
    :type weights: list
    :param selected: selected tensor
    :type selected: list
    """
    for i, matrix in enumerate(scores):
        assert np.array_equal(select(np.array(matrix), np.array(weights)),
                              np.array(selected[i]), equal_nan=True)


def test_select_vector(scores: list, weights: list) -> None:
    """
    test select on vector input
    :param scores: tensor prediction scores
    :type scores: list
    :param weights: vector of model weights
    :type weights: list
    """
    for matrix in scores:
        for vector in matrix:
            with pytest.raises(np.AxisError):
                select(np.array(vector), np.array(weights))


def test_select_scalar(scores: list, weights: list) -> None:
    """
    test select on scalar input
    :param scores: tensor prediction scores
    :type scores: list
    :param weights: vector of model weights
    :type weights: list
    """
    for matrix in scores:
        for vector in matrix:
            for scalar in vector:
                with pytest.raises(np.AxisError):
                    select(np.array(scalar), np.array(weights))
