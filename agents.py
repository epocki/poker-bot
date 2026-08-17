"""Baseline agents. An agent maps a HandState (from its seat's view) to an Action.

These are the placeholder 'brains' we'll eventually replace with a CFR strategy.
They're also useful as sparring partners to benchmark the CFR bot against.
"""

from __future__ import annotations

import random

from engine.state import HandState, Action, CHECK, CALL, FOLD, RAISE


class RandomAgent:
    """Picks uniformly among legal action kinds; random size when raising."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def act(self, s: HandState) -> Action:
        actions = s.legal_actions()
        a = self.rng.choice(actions)
        if a.kind == RAISE:
            lo, hi = s.raise_bounds(s.to_act)
            to = self.rng.randint(lo, hi)
            return Action(RAISE, to)
        return a


class CallingStation:
    """Never folds if it can help it: checks/calls always."""

    def act(self, s: HandState) -> Action:
        kinds = {a.kind for a in s.legal_actions()}
        if CHECK in kinds:
            return Action(CHECK)
        return Action(CALL)
