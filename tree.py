"""Static game-tree precomputation for the preflop game.

The betting tree is identical for every deal (actions never depend on hole cards),
so we enumerate it once into a flat node array. CFR then walks node indices with
no HandState cloning; hole cards enter only as an equity lookup at showdown
terminals. This is what makes training fast.
"""

from __future__ import annotations

from engine.cards import parse_cards
from preflop_game import PreflopGame


class Node:
    __slots__ = ("is_terminal", "player", "history", "tokens", "children",
                 "showdown", "matched", "fold_util_p0")


# Any two non-colliding hands; only used to drive the card-independent betting.
_DUMMY_HOLES = [parse_cards("As Kd"), parse_cards("Qh Jc")]


def build_tree(game: PreflopGame) -> tuple[list[Node], int]:
    nodes: list[Node] = []

    def rec(s, history: str) -> int:
        idx = len(nodes)
        n = Node()
        nodes.append(n)
        if game.is_terminal(s):
            n.is_terminal = True
            n.history = history
            matched = min(s.total_committed)
            if s.folded is not None:
                n.showdown = False
                winner = 1 - s.folded
                n.fold_util_p0 = float(matched) if winner == 0 else float(-matched)
            else:
                n.showdown = True
                n.matched = matched
            return idx
        n.is_terminal = False
        n.player = s.to_act
        n.history = history
        acts = game.actions(s)
        n.tokens = [PreflopGame.token(a) for a in acts]
        n.children = [rec(s.apply(a), history + t) for a, t in zip(acts, n.tokens)]
        return idx

    root = rec(game.root(button=0, holes=_DUMMY_HOLES), "")
    return nodes, root


def count(nodes: list[Node]) -> tuple[int, int]:
    decisions = sum(1 for n in nodes if not n.is_terminal)
    terminals = sum(1 for n in nodes if n.is_terminal)
    return decisions, terminals
