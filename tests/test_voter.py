""" test eFusor voter """

import numpy as np

from efusor.voter import harden, vote


def test_harden(scores: list, hard_scores: list) -> None:
    """
    test harden
    :param scores: prediction scores
    :type scores: list
    :param hard_scores: hardened scores
    :type hard_scores: list
    """
    for i, matrix in enumerate(scores):
        for j, vector in enumerate(matrix):
            assert harden(vector).tolist() == hard_scores[i][j]


def test_hard_voting(scores: list, hard_votes: list) -> None:
    """
    test hard voting (majority)
    :param scores: prediction scores
    :type scores: list
    :param hard_votes: voting results
    :type hard_votes: list
    """
    assert vote(np.array(scores)).tolist() == hard_votes


def test_soft_voting(scores: list, soft_votes: list) -> None:
    """
    test soft voting (average)
    :param scores: prediction scores
    :type scores: list
    :param soft_votes: voting results
    :type soft_votes: list
    """
    assert vote(np.array(scores), method="soft_voting").round(2).tolist() == soft_votes


def test_weighted_hard_voting(scores: list,
                              weights: list,
                              weighted_hard_votes: list
                              ) -> None:
    """
    test weighted hard voting (majority)
    :param scores: prediction scores
    :type scores: list
    :param weights: predictor weights
    :type weights: list
    :param weighted_hard_votes: voting results
    :type weighted_hard_votes: list
    """
    assert vote(np.array(scores), weights=np.array(weights)).tolist() == weighted_hard_votes


def test_weighted_soft_voting(scores: list,
                              weights: list,
                              weighted_soft_votes: list
                              ) -> None:
    """
    test weighted hard voting (majority)
    :param scores: prediction scores
    :type scores: list
    :param weights: predictor weights
    :type weights: list
    :param weighted_soft_votes: voting results
    :type weighted_soft_votes: list
    """
    result = vote(np.array(scores), weights=np.array(weights), method="soft_voting")
    assert result.round(3).tolist() == weighted_soft_votes
