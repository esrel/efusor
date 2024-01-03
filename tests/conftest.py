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
    scores = [
        [[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]],
        [[2.0, 2.0, 1.0], [1.0, 1.0, 2.0]],
        [[2.0, 2.0, 0.0], [1.0, 2.0, 0.0]],
        [[2.0, 2.0, 2.0], [0.0, 0.0, 2.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    ]
    return scores


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


# references
@pytest.fixture(name="borda_result")
def get_borda_result() -> list:
    """
    get borda voting results
    :return: borda result
    :rtype: list
    """
    return [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.5, 3.5, 0.0], [1.0, 1.0, 3.0], [0.0, 0.0, 0.0]]
