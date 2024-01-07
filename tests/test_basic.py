""" test basic fusion """

import pytest

import numpy as np

from efusor.basic import apply


def test_apply_tensor(scores: list, score_max: list) -> None:
    """
    test apply on tensor
    :param scores: prediction scores
    :type scores: list
    :param score_max: references
    :type score_max: list
    """
    assert apply(np.array(scores)).tolist() == score_max


def test_apply_matrix(scores: list, score_max: list) -> None:
    """
    test apply on matrix
    :param scores: prediction scores
    :type scores: list
    :param score_max: references
    :type score_max: list
    """
    for i, matrix in enumerate(scores):
        assert apply(np.array(matrix)).tolist() == score_max[i]


def test_apply_vector(scores: list) -> None:
    """
    test apply on vector
    :param scores: prediction scores
    :type scores: list
    """
    for matrix in scores:
        for vector in matrix:
            # ndim check
            assert np.array_equal(apply(np.array(vector)), np.array(vector), equal_nan=True)


def test_apply_scalar(scores: list) -> None:
    """
    test apply on vector
    :param scores: prediction scores
    :type scores: list
    """
    for matrix in scores:
        for vector in matrix:
            for scalar in vector:
                with pytest.raises(IndexError):
                    apply(np.array(scalar))


def test_apply_tensor_min(scores: list, score_min: list) -> None:
    """
    test apply on tensor
    :param scores: prediction scores
    :type scores: list
    :param score_min: references
    :type score_min: list
    """
    assert apply(np.array(scores), method="min").tolist() == score_min


def test_apply_tensor_sum(scores: list, score_sum: list) -> None:
    """
    test apply on tensor
    :param scores: prediction scores
    :type scores: list
    :param score_sum: references
    :type score_sum: list
    """
    assert apply(np.array(scores), method="sum").round(2).tolist() == score_sum


def test_apply_tensor_product(scores: list, score_prd: list) -> None:
    """
    test apply on tensor
    :param scores: prediction scores
    :type scores: list
    :param score_prd: references
    :type score_prd: list
    """
    assert apply(np.array(scores), method="product").round(2).tolist() == score_prd


def test_apply_tensor_median(scores: list, score_med: list) -> None:
    """
    test apply on tensor
    :param scores: prediction scores
    :type scores: list
    :param score_med: references
    :type score_med: list
    """
    assert apply(np.array(scores), method="median").round(2).tolist() == score_med


def test_apply_tensor_average(scores: list, score_avg: list) -> None:
    """
    test apply on tensor
    :param scores: prediction scores
    :type scores: list
    :param score_avg: references
    :type score_avg: list
    """
    assert apply(np.array(scores), method="average").round(2).tolist() == score_avg
