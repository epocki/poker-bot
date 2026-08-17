"""Vectorized full-width CFR+ for the preflop game.

Instead of sampling one deal per iteration (slow ~1/sqrt(t) convergence), this
handles ALL 169x169 deals in a single tree pass using numpy. Card removal is
handled exactly by the symmetric deal-weight matrix W applied at terminals, so
the correlation between the two players' hands is preserved.

Per-class regrets/strategy-sums are numpy arrays of shape (169, n_actions) at each
decision node. Terminal counterfactual values reduce to matrix-vector products
with W and A = W*EQ. CFR+ (regret flooring + linear averaging) gives ~1/t
convergence, so a near-equilibrium solve takes a few thousand iterations.
"""

from __future__ import annotations

import numpy as np

from engine.equity import CLASSES, N_CLASSES
from preflop_game import PreflopGame
from tree import build_tree
from exploitability import deal_weights


class VectorTrainer:
    def __init__(self, game: PreflopGame) -> None:
        self.game = game
        self.nodes, self.root = build_tree(game)
        n = N_CLASSES

        self.EQ = np.array(game.eq.table, dtype=np.float64)      # (n,n) P0 equity
        self.W = np.array(deal_weights(), dtype=np.float64)      # (n,n) deal probs
        self.A = self.W * self.EQ                                # elementwise

        # Per decision-node parameters.
        self.regret = [None] * len(self.nodes)
        self.strat_sum = [None] * len(self.nodes)
        for i, nd in enumerate(self.nodes):
            if not nd.is_terminal:
                na = len(nd.tokens)
                self.regret[i] = np.zeros((n, na))
                self.strat_sum[i] = np.zeros((n, na))

    def _sigma(self, i: int) -> np.ndarray:
        r = self.regret[i]
        pos = np.maximum(r, 0.0)
        tot = pos.sum(axis=1, keepdims=True)
        na = r.shape[1]
        # uniform where no positive regret yet
        out = np.where(tot > 0, pos / np.where(tot > 0, tot, 1.0), 1.0 / na)
        return out

    def _cfr(self, nidx, r0, r1, w):
        n = self.nodes[nidx]
        W, A = self.W, self.A
        if n.is_terminal:
            if n.showdown:
                m = n.matched
                cfv0 = m * (2.0 * (A @ r1) - (W @ r1))
                cfv1 = m * (2.0 * (A @ r0) - (W @ r0))
                return cfv0, cfv1
            k = n.fold_util_p0
            return k * (W @ r1), -k * (W @ r0)

        p = n.player
        sigma = self._sigma(nidx)               # (n, na)
        na = sigma.shape[1]
        rp = r0 if p == 0 else r1
        self.strat_sum[nidx] += w * (rp[:, None] * sigma)

        cfv0_acc = np.zeros(N_CLASSES)
        cfv1_acc = np.zeros(N_CLASSES)
        cfv_p_actions = np.zeros((N_CLASSES, na))
        for a in range(na):
            if p == 0:
                c0v, c1v = self._cfr(n.children[a], r0 * sigma[:, a], r1, w)
                cfv1_acc += c1v
                cfv_p_actions[:, a] = c0v
            else:
                c0v, c1v = self._cfr(n.children[a], r0, r1 * sigma[:, a], w)
                cfv0_acc += c0v
                cfv_p_actions[:, a] = c1v

        cfv_p_node = (sigma * cfv_p_actions).sum(axis=1)   # (n,)
        # CFR+ regret update (counterfactual reach already baked into cfv values)
        delta = cfv_p_actions - cfv_p_node[:, None]
        self.regret[nidx] = np.maximum(self.regret[nidx] + delta, 0.0)

        if p == 0:
            return cfv_p_node, cfv1_acc
        return cfv0_acc, cfv_p_node

    def train(self, iterations: int, log_every: int = 0) -> None:
        ones = np.ones(N_CLASSES)
        for t in range(1, iterations + 1):
            self._cfr(self.root, ones, ones, float(t))
            if log_every and t % log_every == 0:
                print(f"  iter {t:,}")

    def average_strategy(self) -> dict:
        out = {}
        for i, nd in enumerate(self.nodes):
            if nd.is_terminal:
                continue
            ss = self.strat_sum[i]
            tot = ss.sum(axis=1, keepdims=True)
            avg = np.where(tot > 0, ss / np.where(tot > 0, tot, 1.0), 0.0)
            for ci in range(N_CLASSES):
                if tot[ci, 0] > 0:
                    key = CLASSES[ci] + ":" + nd.history
                    out[key] = {t: float(avg[ci, k]) for k, t in enumerate(nd.tokens)}
        return out

    # convenience so exploitability.exploitability_bb100 works unchanged
    @property
    def eq(self):
        return self.game.eq.table
