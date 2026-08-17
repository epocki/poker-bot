"""Preflop-only NLHE game for CFR.

Reuses the tested HandState betting engine but (a) restricts raises to a small
bet-size abstraction and (b) ends the hand when preflop betting completes,
scoring the pot by all-in equity instead of dealing a board. This is the exact
game the CFR solver traverses.

Terminal conditions:
  * a fold                          -> folder loses the matched amount
  * preflop betting completes       -> pot split by equity of the two hands
  * an all-in is called             -> pot split by equity (same as above)

Everything is measured in chips; divide by the big blind for bb units.
"""

from __future__ import annotations

from engine.cards import Card, parse_cards
from engine.state import HandState, Action, FOLD, CHECK, CALL, RAISE, PREFLOP
from engine.equity import EquityTable, card_class

# The board is never used for preflop terminals (equity handles the runout), but
# HandState requires a 5-card board at construction. This placeholder is never
# evaluated.
_DUMMY_BOARD: list[Card] = parse_cards("2c 3d 7h 9s Tc")


class PreflopGame:
    def __init__(self, eq: EquityTable, sb: int = 1, bb: int = 2,
                 stack: int = 200, raise_fractions=(0.5, 1.0),
                 no_open_limp: bool = False) -> None:
        self.eq = eq
        self.sb = sb
        self.bb = bb
        self.stack = stack                 # starting stack in chips (100bb = 200)
        self.raise_fractions = tuple(raise_fractions)
        # If True, the small blind may not complete the blind as its opening
        # action (no limping); it must raise or fold first-in. Calling a raise
        # later is unaffected.
        self.no_open_limp = no_open_limp

    def root(self, button: int, holes: list[list[Card]]) -> HandState:
        return HandState.new_hand(self.sb, self.bb, button,
                                  [self.stack, self.stack], holes, _DUMMY_BOARD)

    # ---- terminal handling ----

    def is_terminal(self, s: HandState) -> bool:
        # Folded / all-in-runout finish HandState; a completed preflop round
        # advances the street past PREFLOP. Either way, we stop here.
        return s.finished or s.street > PREFLOP

    def terminal_util_p0(self, s: HandState) -> float:
        """Utility to player 0 (chips), zero-sum so player 1 gets the negation."""
        matched = min(s.total_committed)
        if s.folded is not None:
            winner = 1 - s.folded
            return float(matched) if winner == 0 else float(-matched)
        # Showdown by equity on the matched pot.
        eq0 = self.eq.equity(card_class(*s.holes[0]), card_class(*s.holes[1]))
        pot = 2 * matched
        return eq0 * pot - matched          # = matched * (2*eq0 - 1)

    # ---- action abstraction ----

    def actions(self, s: HandState) -> list[Action]:
        """Legal actions with raises restricted to the size abstraction."""
        p = s.to_act
        base = s.legal_actions()
        # Drop the opening limp (SB completing an unraised pot) when disabled.
        opening_limp = s.street == PREFLOP and s.current_bet == self.bb
        out = [a for a in base
               if a.kind in (FOLD, CHECK, CALL)
               and not (self.no_open_limp and opening_limp and a.kind == CALL)]

        lo, hi = s.raise_bounds(p)
        if lo is None:
            return out
        pot_now = sum(s.total_committed)
        to_call = s.to_call(p)
        sizes: set[int] = set()
        for fr in self.raise_fractions:
            raw = s.current_bet + round(fr * (pot_now + to_call))
            sizes.add(max(lo, min(hi, raw)))
        sizes.add(hi)                       # always offer all-in
        for size in sorted(sizes):
            out.append(Action(RAISE, size))
        return out

    @staticmethod
    def token(a: Action) -> str:
        """Compact history token for infoset keys."""
        if a.kind == FOLD:
            return "f"
        if a.kind == CHECK:
            return "x"
        if a.kind == CALL:
            return "c"
        return f"r{a.to_amount}"
