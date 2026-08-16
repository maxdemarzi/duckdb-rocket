"""The routing rule has to escalate the rows it says it escalates.

Every number in the feature-routing table is an accuracy after swapping a chosen subset of rows, so
a rule that picked the wrong subset -- the most confident rather than the least, or the wrong count
-- would still produce a plausible table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from feature_route import margins, route  # noqa: E402


def test_margins_are_top1_minus_top2():
    p = np.array([[0.7, 0.2, 0.1], [0.4, 0.35, 0.25], [1.0, 0.0, 0.0]])
    assert np.allclose(margins(p), [0.5, 0.05, 1.0])


def test_a_zero_budget_is_the_primary_alone():
    primary = np.array([[0.9, 0.1], [0.2, 0.8]])
    alternate = np.array([[0.1, 0.9], [0.9, 0.1]])
    assert route(primary, alternate, 0.0).tolist() == [0, 1]


def test_a_full_budget_is_the_alternate_alone():
    primary = np.array([[0.9, 0.1], [0.2, 0.8]])
    alternate = np.array([[0.1, 0.9], [0.9, 0.1]])
    assert route(primary, alternate, 1.0).tolist() == [1, 0]


def test_the_least_confident_rows_are_the_ones_escalated():
    """Row 1 has the smallest margin, so a 25% budget on four rows must take exactly it."""
    primary = np.array([[0.95, 0.05],      # margin 0.90
                        [0.51, 0.49],      # margin 0.02  <- least confident
                        [0.80, 0.20],      # margin 0.60
                        [0.70, 0.30]])     # margin 0.40
    alternate = np.array([[0.0, 1.0]] * 4)  # always predicts class 1
    out = route(primary, alternate, 0.25)
    assert out.tolist() == [0, 1, 0, 0], "the escalated row was not the least confident one"


def test_the_budget_is_a_row_count_not_a_threshold():
    primary = np.array([[0.6, 0.4]] * 10)
    alternate = np.array([[0.0, 1.0]] * 10)
    for budget, expected in ((0.1, 1), (0.3, 3), (0.5, 5)):
        assert int((route(primary, alternate, budget) == 1).sum()) == expected


def test_routing_does_not_mutate_its_input():
    """`out` starts as the primary's argmax; writing through a view would corrupt the next budget."""
    primary = np.array([[0.95, 0.05], [0.51, 0.49]])
    alternate = np.array([[0.0, 1.0], [0.0, 1.0]])
    before = primary.copy()
    route(primary, alternate, 0.5)
    assert np.array_equal(primary, before)
    # And a second call at budget 0 must still see the untouched primary.
    assert route(primary, alternate, 0.0).tolist() == [0, 0]


def test_majority_vote_is_a_plurality_with_a_stated_tie_break():
    from feature_route import majority_vote
    # preds is (arms, rows), so read the COLUMNS: row 0 unanimous for class 0, row 1 a 3-1 split
    # for class 1, row 2 a 2-2 tie between classes 0 and 1.
    preds = np.array([[0, 1, 0],
                      [0, 1, 0],
                      [0, 1, 1],
                      [0, 0, 1]])
    assert majority_vote(preds, 2).tolist() == [0, 1, 0], "the tie did not go to the lowest index"


def test_majority_vote_counts_arms_not_probabilities():
    """Three arms weakly agreeing must beat one arm that is certain -- that is the point of a vote.

    An averaging rule gets the opposite answer here, which is why both are reported.
    """
    from feature_route import majority_vote
    preds = np.array([[1], [1], [1], [0]])
    assert majority_vote(preds, 2).tolist() == [1]
