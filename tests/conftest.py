""" eFusor Test Cases """

import pytest

import numpy as np


@pytest.fixture(name="scores")
def get_preds_scores() -> list:
    """
    get prediction scores
    :return: tensor
    :rtype: list
    """
    return [
        [[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]],  # proper in different orders
        [[0.4, 0.4, 0.2], [0.2, 0.2, 0.6]],  # tie on 1st & tie on 2nd
        [[0.4, 0.4, 0.0], [0.2, 0.7, 0.0]],  # tie on 1st with 0 & proper with 0
        [[0.3, 0.3, 0.3], [0.0, 0.0, 1.0]],  # tie on all 3 & single prediction
        [[np.nan, np.nan, np.nan], [0.0, 0.0, 0.0]]  # nan & 0.0
    ]


@pytest.fixture(name="weights")
def get_weights() -> list:
    """
    get prediction weights (vector)
    :return: weights
    :rtype: list
    """
    return [0.75, 0.25]


# transformations
@pytest.fixture(name="borda_scores")
def get_borda_scores() -> list:
    """
    get borda scores
    :return: score tensor
    :rtype: list
    """
    return [
        [[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]],
        [[1.5, 1.5, 0.0], [0.5, 0.5, 2.0]],
        [[1.5, 1.5, 0.0], [1.0, 2.0, 0.0]],
        [[1.0, 1.0, 1.0], [0.0, 0.0, 2.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    ]


@pytest.fixture(name="borda_simple")
def get_simple_scores():
    """
    get borda simple scores
    :return: simple scores
    :rtype: list
    """
    return [
        [[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]],
        [[2.0, 2.0, 1.0], [1.0, 1.0, 2.0]],
        [[2.0, 2.0, 0.0], [1.0, 2.0, 0.0]],
        [[2.0, 2.0, 2.0], [0.0, 0.0, 2.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    ]


@pytest.fixture(name="ranks")
def get_ranks() -> list:
    """
    get test ranks
    :return: rank tensor
    :rtype: list
    """
    return [
        [[2, 1, 0], [0, 1, 2]],
        [[0, 0, 1], [1, 1, 0]],
        [[0, 0, 1], [1, 0, 2]],
        [[0, 0, 0], [1, 1, 0]],
        [[0, 0, 0], [0, 0, 0]]
    ]


@pytest.fixture(name="norms")
def get_norms() -> list:
    """
    get test min-max norms (need to be rounded!)
    :return: norm tensor
    :rtype: list
    """
    return [
        [[0.00, 0.33, 1.00], [1.00, 0.17, 0.00]],
        [[1.00, 1.00, 0.00], [0.00, 0.00, 1.00]],
        [[1.00, 1.00, 0.00], [0.29, 1.00, 0.00]],
        [[0.30, 0.30, 0.30], [0.00, 0.00, 1.00]],
        [[np.nan, np.nan, np.nan], [0.00, 0.00, 0.00]]
    ]


@pytest.fixture(name="hard_scores")
def get_hardened() -> list:
    """
    get hardened scores
    :return: hardened tensor
    :rtype: list
    """
    return [
        [[0, 0, 1], [1, 0, 0]],
        [[1, 1, 0], [0, 0, 1]],
        [[1, 1, 0], [0, 1, 0]],
        [[1, 1, 1], [0, 0, 1]],
        [[0, 0, 0], [0, 0, 0]]
    ]


# references
@pytest.fixture(name="hard_votes")
def get_hard_votes() -> list:
    """
    get hard voting results
    :return: hard votes
    :rtype: list
    """
    return [[1, 0, 1], [1, 1, 1], [1, 2, 0], [1, 1, 2], [0, 0, 0]]


@pytest.fixture(name="soft_votes")
def get_soft_votes() -> list:
    """
    get soft voting results
    :return: soft votes
    :rtype: list
    """
    return [[0.45, 0.25, 0.30],
            [0.30, 0.30, 0.40],
            [0.30, 0.55, 0.00],
            [0.15, 0.15, 0.65],
            [0.00, 0.00, 0.00]]


@pytest.fixture(name="weighted_hard_votes")
def get_weighted_hard_votes() -> list:
    """
    get weighted hard voting results for [0.75, 0.25]
    :return: hard votes
    :rtype: list
    """
    return [[0.25, 0.00, 0.75],
            [0.75, 0.75, 0.25],
            [0.75, 1.00, 0.00],
            [0.75, 0.75, 1.00],
            [0.00, 0.00, 0.00]]


@pytest.fixture(name="weighted_soft_votes")
def get_weighted_soft_votes() -> list:
    """
    get weighted soft voting results for [0.75, 0.25]
    :return: soft votes
    :rtype: list
    """
    return [[0.325, 0.275, 0.400],
            [0.350, 0.350, 0.300],
            [0.350, 0.475, 0.000],
            [0.225, 0.225, 0.475],
            [0.000, 0.000, 0.000]]


@pytest.fixture(name="borda_result")
def get_borda_result() -> list:
    """
    get borda voting results
    :return: borda result
    :rtype: list
    """
    return [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.5, 3.5, 0.0], [1.0, 1.0, 3.0], [0.0, 0.0, 0.0]]


# basic operations
@pytest.fixture(name="score_max")
def get_basic_max() -> list:
    """
    get max of scores
    :return: max fusion
    :rtype: list
    """
    return [[0.7, 0.3, 0.5], [0.4, 0.4, 0.6], [0.4, 0.7, 0.0], [0.3, 0.3, 1.0], [0.0, 0.0, 0.0]]


@pytest.fixture(name="score_min")
def get_basic_min() -> list:
    """
    get min of scores
    :return: min fusion
    :rtype: list
    """
    return [[0.2, 0.2, 0.1], [0.2, 0.2, 0.2], [0.2, 0.4, 0.0], [0.0, 0.0, 0.3], [0.0, 0.0, 0.0]]


@pytest.fixture(name="score_sum")
def get_basic_sum() -> list:
    """
    get sum of scores (need to be rounded!)
    :return: sum fusion
    :rtype: list
    """
    return [[0.57, 0.17, 0.27],
            [0.27, 0.27, 0.47],
            [0.27, 0.77, 0.00],
            [0.00, 0.00, 0.97],
            [0.00, 0.00, 0.00]]


@pytest.fixture(name="score_prd")
def get_basic_product() -> list:
    """
    get product of scores (need to be rounded!)
    :return: product fusion
    :rtype: list
    """
    return [[0.05, 0.02, 0.02],
            [0.03, 0.03, 0.04],
            [0.03, 0.09, 0.00],
            [0.00, 0.00, 0.10],
            [0.00, 0.00, 0.00]]


@pytest.fixture(name="score_med")
def get_basic_median() -> list:
    """
    get median of scores
    :return: median fusion
    :rtype: list
    """
    return [[0.45, 0.25, 0.30],
            [0.30, 0.30, 0.40],
            [0.30, 0.55, 0.00],
            [0.15, 0.15, 0.65],
            [0.00, 0.00, 0.00]]


@pytest.fixture(name="score_avg")
def get_basic_average() -> list:
    """
    get average of scores (same as median)
    :return: average fusion
    :rtype: list
    """
    return [[0.45, 0.25, 0.30],
            [0.30, 0.30, 0.40],
            [0.30, 0.55, 0.00],
            [0.15, 0.15, 0.65],
            [0.00, 0.00, 0.00]]


# decisor
@pytest.fixture(name="re_ranked")
def get_reranking() -> list:
    """
    get scores re-ranked by weights
    :return: re-ranked scores
    :rtype: list
    """
    # weights [0.75, 0.25]
    return [
        [[0.150, 0.225, 0.375], [0.175, 0.050, 0.025]],
        [[0.300, 0.300, 0.150], [0.050, 0.050, 0.150]],
        [[0.300, 0.300, 0.000], [0.050, 0.175, 0.000]],
        [[0.225, 0.225, 0.225], [0.000, 0.000, 0.250]],
        [[np.nan, np.nan, np.nan], [0.000, 0.000, 0.000]]
    ]


@pytest.fixture(name="selected")
def get_selection() -> list:
    """
    get slice by weights argmax
    :return: selected prediction scores
    :rtype: list
    """
    # weights [0.75, 0.25]
    return [[0.2, 0.3, 0.5],
            [0.4, 0.4, 0.2],
            [0.4, 0.4, 0.0],
            [0.3, 0.3, 0.3],
            [np.nan, np.nan, np.nan]]


@pytest.fixture(name="prioritized")
def get_prioritized() -> list:
    """
    get slice by weights ranks
    :return: sliced prediction scores
    :rtype: list
    """
    # weights [0.75, 0.25]
    return [[0.2, 0.3, 0.5],
            [0.4, 0.4, 0.2],
            [0.4, 0.4, 0.0],
            [0.3, 0.3, 0.3],
            [0.0, 0.0, 0.0]]


@pytest.fixture(name="incremented")
def get_incremented() -> list:
    """
    get max of rank-incremented scores
    :return: fused prediction scores
    :rtype: list
    """
    # weights [0.75, 0.25] -> addends [2.0, 1.0]
    return [[2.2, 2.3, 2.5],
            [2.4, 2.4, 2.2],
            [2.4, 2.4, 2.0],
            [2.3, 2.3, 2.3],
            [1.0, 1.0, 1.0]]
