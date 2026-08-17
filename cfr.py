"""CFR+ solver for the preflop game, operating on the precomputed static tree.

Two-player zero-sum CFR tracking utility from player 0's perspective. Chance (the
deal) is Monte-Carlo sampled each iteration by shuffling a real deck, which
reproduces the correct hand distribution *and* card removal. CFR+ variant:
regrets floored at zero, average strategy accumulated with linear (iteration-
weighted) averaging.

Button is fixed to seat 0 (player 0 = small blind/button, player 1 = big blind);
heads-up symmetry makes this fully general.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict

from engine.cards import make_deck
from engine.equity import EquityTable, CLASSES, CLASS_INDEX, card_class
from preflop_game import PreflopGame
from tree import build_tree, Node


class Trainer:
    def __init__(self, game: PreflopGame) -> None:
        self.game = game
        self.nodes, self.root = build_tree(game)
        self.eq = game.eq.table            # raw 2D list, indexed by class index
        self.regret: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.strat_sum: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def _strategy(self, key: str, tokens: list[str]) -> dict[str, float]:
        r = self.regret[key]
        pos = {t: (r[t] if r[t] > 0 else 0.0) for t in tokens}
        total = sum(pos.values())
        if total > 0:
            return {t: pos[t] / total for t in tokens}
        u = 1.0 / len(tokens)
        return {t: u for t in tokens}

    def _cfr(self, nidx: int, i0: int, i1: int, p0r: float, p1r: float, w: int) -> float:
        n = self.nodes[nidx]
        if n.is_terminal:
            if n.showdown:
                eq0 = self.eq[i0][i1]
                return n.matched * (2.0 * eq0 - 1.0)
            return n.fold_util_p0

        p = n.player
        cls = CLASSES[i0] if p == 0 else CLASSES[i1]
        key = cls + ":" + n.history
        tokens = n.tokens
        sigma = self._strategy(key, tokens)

        reach_p = p0r if p == 0 else p1r
        ss = self.strat_sum[key]
        for t in tokens:
            ss[t] += w * reach_p * sigma[t]

        child = {}
        node_util = 0.0
        for t, cidx in zip(tokens, n.children):
            if p == 0:
                cu = self._cfr(cidx, i0, i1, p0r * sigma[t], p1r, w)
            else:
                cu = self._cfr(cidx, i0, i1, p0r, p1r * sigma[t], w)
            child[t] = cu
            node_util += sigma[t] * cu

        cf_reach = p1r if p == 0 else p0r
        reg = self.regret[key]
        sign = 1.0 if p == 0 else -1.0
        node_for_p = sign * node_util
        for t in tokens:
            reg[t] = max(0.0, reg[t] + cf_reach * (sign * child[t] - node_for_p))
        return node_util

    def train(self, iterations: int, seed: int = 0, log_every: int = 0) -> None:
        rng = random.Random(seed)
        deck = make_deck()
        idx = CLASS_INDEX
        for t in range(1, iterations + 1):
            rng.shuffle(deck)
            i0 = idx[card_class(deck[0], deck[1])]
            i1 = idx[card_class(deck[2], deck[3])]
            self._cfr(self.root, i0, i1, 1.0, 1.0, t)
            if log_every and t % log_every == 0:
                print(f"  iter {t:,}  infosets={len(self.strat_sum):,}")

    def average_strategy(self) -> dict[str, dict[str, float]]:
        out = {}
        for key, ss in self.strat_sum.items():
            total = sum(ss.values())
            if total > 0:
                out[key] = {t: v / total for t, v in ss.items()}
        return out

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.average_strategy(), f)
