""" test basic fusion """

import numpy as np

from efusor.basic import apply


def test_apply_tensor_max(scores: list, score_max: list) -> None:
    """
    test apply on tensor
    :param scores: prediction scores
    :type scores: list
    :param score_max: references
    :type score_max: list
    """
    assert apply(np.array(scores), method="max").tolist() == score_max


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
