"""Dealing / chance. Shuffles a deck and produces holes + full board.

Kept separate from HandState so the same primitive serves both full-hand
simulation and (later) CFR's chance sampling. Deterministic given a seed.
"""

from __future__ import annotations

import random

from .cards import Card, make_deck


class Dealer:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def deal(self) -> tuple[list[list[Card]], list[Card]]:
        """Return (holes, board): holes[player] = [Card, Card], board = 5 cards."""
        deck = make_deck()
        self.rng.shuffle(deck)
        holes = [[deck[0], deck[2]], [deck[1], deck[3]]]  # alternate as in a real deal
        board = deck[4:9]
        return holes, board
