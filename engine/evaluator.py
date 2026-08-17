"""Poker hand evaluation.

`evaluate` returns a comparable tuple key for any 5-7 card hand: larger is
stronger, and keys are directly comparable across all hand types. The key is
(category, tiebreakers...) where category is one of the HandCategory ints.

Design choice: we score the best 5-card hand out of the given cards by checking
all C(n,5) combinations against a correct 5-card scorer. This is O(21) for a
7-card hand — trivially fast for engine work and testing. If the CFR solver
later needs millions of evals/sec we can swap in a lookup-table evaluator behind
this same interface, but correctness comes first.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

from .cards import Card

# Hand categories, ordered weakest (0) to strongest (8).
HIGH_CARD = 0
PAIR = 1
TWO_PAIR = 2
TRIPS = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
QUADS = 7
STRAIGHT_FLUSH = 8

CATEGORY_NAMES = {
    HIGH_CARD: "high card",
    PAIR: "pair",
    TWO_PAIR: "two pair",
    TRIPS: "three of a kind",
    STRAIGHT: "straight",
    FLUSH: "flush",
    FULL_HOUSE: "full house",
    QUADS: "four of a kind",
    STRAIGHT_FLUSH: "straight flush",
}


def _straight_high(unique_ranks: list[int]) -> int | None:
    """Given the distinct ranks present (any order), return the high card of the
    best straight, or None. Handles the wheel (A-2-3-4-5) where the ace plays low
    and the straight's high card is 5.
    """
    rset = set(unique_ranks)
    # Ace plays low for the wheel: treat a 14 as also a 1.
    if 14 in rset:
        rset.add(1)
    ordered = sorted(rset, reverse=True)
    run = 1
    for i in range(1, len(ordered)):
        if ordered[i] == ordered[i - 1] - 1:
            run += 1
            if run >= 5:
                return ordered[i] + 4  # high card of this 5-run
        else:
            run = 1
    return None


def score5(cards: list[Card]) -> tuple:
    """Score exactly 5 cards. Returns a comparable (category, *tiebreakers) key."""
    if len(cards) != 5:
        raise ValueError(f"score5 needs exactly 5 cards, got {len(cards)}")

    ranks = [c.rank for c in cards]
    suits = [c.suit for c in cards]
    is_flush = len(set(suits)) == 1

    # Rank multiplicities, ordered by (count desc, rank desc) so the primary
    # group leads the tiebreak vector.
    counts = Counter(ranks)
    by_group = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    shape = tuple(cnt for _, cnt in by_group)           # e.g. (3, 2) for a boat
    ordered_ranks = tuple(rank for rank, _ in by_group)  # ranks in tiebreak order

    straight_high = _straight_high(list(counts.keys())) if len(counts) == 5 else None

    if is_flush and straight_high is not None:
        return (STRAIGHT_FLUSH, straight_high)
    if shape == (4, 1):
        return (QUADS, *ordered_ranks)
    if shape == (3, 2):
        return (FULL_HOUSE, *ordered_ranks)
    if is_flush:
        return (FLUSH, *sorted(ranks, reverse=True))
    if straight_high is not None:
        return (STRAIGHT, straight_high)
    if shape == (3, 1, 1):
        return (TRIPS, *ordered_ranks)
    if shape == (2, 2, 1):
        return (TWO_PAIR, *ordered_ranks)
    if shape == (2, 1, 1, 1):
        return (PAIR, *ordered_ranks)
    return (HIGH_CARD, *sorted(ranks, reverse=True))


def evaluate(cards: list[Card]) -> tuple:
    """Score the best 5-card hand from 5, 6, or 7 cards. Larger key = stronger."""
    n = len(cards)
    if n < 5 or n > 7:
        raise ValueError(f"evaluate needs 5-7 cards, got {n}")
    if len(set(cards)) != n:
        raise ValueError("duplicate cards passed to evaluate")
    if n == 5:
        return score5(cards)
    return max(score5(list(combo)) for combo in combinations(cards, 5))


def category_of(key: tuple) -> int:
    """Extract the HandCategory int from an evaluate/score5 key."""
    return key[0]


def evaluate7_fast(cards: list[Card]) -> tuple:
    """Direct 7-card scorer (no C(7,5) loop). Same key semantics as `evaluate`.

    This is the hot-path evaluator used for building equity tables and CFR
    rollouts. It is differentially tested against `evaluate` (the simple
    reference) to guarantee identical results.
    """
    ranks = [c.rank for c in cards]
    suits = [c.suit for c in cards]
    rank_count = Counter(ranks)
    suit_count = Counter(suits)

    flush_suit = None
    for s, c in suit_count.items():
        if c >= 5:
            flush_suit = s
            break

    # Straight flush (must be checked before everything else).
    if flush_suit is not None:
        flush_ranks = [c.rank for c in cards if c.suit == flush_suit]
        sf_high = _straight_high(flush_ranks)
        if sf_high is not None:
            return (STRAIGHT_FLUSH, sf_high)

    # Group ranks by (count desc, rank desc).
    by_group = sorted(rank_count.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    trips = [r for r, c in by_group if c == 3]
    pairs = [r for r, c in by_group if c == 2]

    if by_group[0][1] == 4:
        quad = by_group[0][0]
        kicker = max(r for r in ranks if r != quad)
        return (QUADS, quad, kicker)

    if trips:
        if len(trips) >= 2:            # two sets -> higher is trips, lower plays as pair
            return (FULL_HOUSE, trips[0], trips[1])
        if pairs:
            return (FULL_HOUSE, trips[0], pairs[0])

    if flush_suit is not None:
        top5 = sorted((c.rank for c in cards if c.suit == flush_suit), reverse=True)[:5]
        return (FLUSH, *top5)

    straight_high = _straight_high(list(rank_count.keys()))
    if straight_high is not None:
        return (STRAIGHT, straight_high)

    if trips:
        t = trips[0]
        kick = sorted((r for r in ranks if r != t), reverse=True)[:2]
        return (TRIPS, t, *kick)

    if len(pairs) >= 2:
        hi, lo = pairs[0], pairs[1]
        kicker = max(r for r in ranks if r != hi and r != lo)
        return (TWO_PAIR, hi, lo, kicker)

    if pairs:
        p = pairs[0]
        kick = sorted((r for r in ranks if r != p), reverse=True)[:3]
        return (PAIR, p, *kick)

    return (HIGH_CARD, *sorted(ranks, reverse=True)[:5])
