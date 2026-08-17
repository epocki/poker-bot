"""Preflop hand-class equity (all-in, full 5-card runout).

The 169 canonical starting-hand classes ("AA", "AKs", "AKo", ...) are the private
information in the preflop game. This module builds a 169x169 table where
table[i][j] = P(class i beats class j) + 0.5 * P(tie), i.e. the equity of hand i
against hand j when all-in preflop, respecting card removal (two players can't
hold the same card).

The table is built by Monte Carlo (shared board deals across all matchups) and
cached to disk, since it only depends on the rules of poker, not on our strategy.
"""

from __future__ import annotations

import json
import os
import random

from .cards import Card, RANKS, make_deck
from .evaluator import evaluate7_fast

RANK_CHARS = "23456789TJQKA"
CACHE_PATH = os.path.join(os.path.dirname(__file__), "equity_table.json")


def _all_classes() -> list[str]:
    """169 canonical class keys, ordered high-rank-first for readability."""
    ranks_desc = sorted(RANKS, reverse=True)  # 14..2
    from .cards import RANK_TO_CHAR
    classes = []
    for i, r1 in enumerate(ranks_desc):
        for j, r2 in enumerate(ranks_desc):
            c1, c2 = RANK_TO_CHAR[r1], RANK_TO_CHAR[r2]
            if r1 == r2:
                if i == j:
                    classes.append(c1 + c2)          # pair, once
            elif r1 > r2:
                classes.append(c1 + c2 + "s")        # suited
                classes.append(c1 + c2 + "o")        # offsuit
    return classes


CLASSES = _all_classes()
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)  # 169


def card_class(a: Card, b: Card) -> str:
    """Canonical class key for two hole cards."""
    from .cards import RANK_TO_CHAR
    hi, lo = (a, b) if a.rank >= b.rank else (b, a)
    if hi.rank == lo.rank:
        return RANK_TO_CHAR[hi.rank] * 2
    suited = "s" if a.suit == b.suit else "o"
    return RANK_TO_CHAR[hi.rank] + RANK_TO_CHAR[lo.rank] + suited


def combos_for_class(key: str) -> list[tuple[Card, Card]]:
    """All concrete 2-card combinations belonging to a class key."""
    from .cards import CHAR_TO_RANK, SUITS
    if len(key) == 2:  # pair, e.g. "AA"
        r = CHAR_TO_RANK[key[0]]
        return [(Card(r, s1), Card(r, s2))
                for a, s1 in enumerate(SUITS) for s2 in SUITS[a + 1:]]
    r1, r2, kind = CHAR_TO_RANK[key[0]], CHAR_TO_RANK[key[1]], key[2]
    if kind == "s":
        return [(Card(r1, s), Card(r2, s)) for s in SUITS]
    return [(Card(r1, s1), Card(r2, s2))
            for s1 in SUITS for s2 in SUITS if s1 != s2]


class EquityTable:
    def __init__(self, table: list[list[float]]) -> None:
        self.table = table

    def equity(self, class_a: str, class_b: str) -> float:
        """Equity of class_a vs class_b (P0 perspective, ties count 0.5)."""
        return self.table[CLASS_INDEX[class_a]][CLASS_INDEX[class_b]]

    def equity_cards(self, a: tuple[Card, Card], b: tuple[Card, Card]) -> float:
        return self.equity(card_class(*a), card_class(*b))

    # ---- persistence ----

    def save(self, path: str = CACHE_PATH) -> None:
        with open(path, "w") as f:
            json.dump({"classes": CLASSES, "table": self.table}, f)

    @classmethod
    def load(cls, path: str = CACHE_PATH) -> "EquityTable":
        with open(path) as f:
            data = json.load(f)
        if data["classes"] != CLASSES:
            raise ValueError("cached equity table has mismatched class ordering")
        return cls(data["table"])

    @classmethod
    def load_or_build(cls, samples: int = 2000, seed: int = 7,
                      path: str = CACHE_PATH) -> "EquityTable":
        if os.path.exists(path):
            return cls.load(path)
        t = build_equity_table(samples, seed)
        t.save(path)
        return t


def build_equity_table(samples_per_matchup: int = 2000, seed: int = 7) -> EquityTable:
    """Stratified Monte Carlo: every one of the ~14k class matchups gets the same
    number of samples, so accuracy is uniform (no rare-matchup starvation). For
    matchup (i, j) we draw random concrete combos of each class (rejecting card
    collisions) and a random board from the remaining deck.
    """
    rng = random.Random(seed)
    n = N_CLASSES
    combos = [combos_for_class(c) for c in CLASSES]
    full_deck = make_deck()
    table = [[0.5] * n for _ in range(n)]

    for i in range(n):
        ci = combos[i]
        for j in range(i, n):
            cj = combos[j]
            acc = 0.0
            for _ in range(samples_per_matchup):
                # draw non-colliding combos for the two classes
                while True:
                    a = ci[rng.randrange(len(ci))]
                    b = cj[rng.randrange(len(cj))]
                    used = {a[0], a[1], b[0], b[1]}
                    if len(used) == 4:
                        break
                while True:                       # reject boards touching hole cards
                    board = rng.sample(full_deck, 5)
                    if used.isdisjoint(board):
                        break
                k0 = evaluate7_fast([a[0], a[1], *board])
                k1 = evaluate7_fast([b[0], b[1], *board])
                acc += 1.0 if k0 > k1 else (0.5 if k0 == k1 else 0.0)
            e = acc / samples_per_matchup
            table[i][j] = e
            table[j][i] = 1.0 - e
    return EquityTable(table)
