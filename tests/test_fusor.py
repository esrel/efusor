""" test eFusor fuse """

from efusor.fusor import fuse, vectorize


def test_fuse(scores: list, weights: list, weighted_hard_votes: list) -> None:
    """
    test fuse wrapper
    :param scores: prediction scores
    :type scores: list
    :param weights: predictor weights
    :type weights: list
    :param weighted_hard_votes: voting decision
    :type weighted_hard_votes: list
    """
    result = fuse(vectorize(scores), method="hard_voting", weights=vectorize(weights))

    assert result.tolist() == weighted_hard_votes
