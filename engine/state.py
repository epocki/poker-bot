"""Heads-up No-Limit Hold'em betting state machine.

This is the core game tree. It is deliberately card-agnostic for the *betting*
logic: hole cards and the full 5-card board are provided up front (dealt by a
Dealer or, later, sampled by CFR's chance step), and the engine progresses
through streets and settles at terminal. Cards only matter at showdown.

Heads-up rules encoded here (the ones people get wrong):
  * The BUTTON posts the small blind and acts FIRST preflop; the big blind acts
    last preflop and has the option to raise a limp.
  * Postflop the button acts LAST (the non-button/big-blind acts first).
  * Min-raise = size of the last full raise (preflop that's the big blind).
  * An all-in raise for LESS than a full raise does NOT reopen betting: the
    player it's shoved over may only call or fold.
  * Uncalled chips are always returned; only matched chips form the pot. In
    heads-up the matched pot is always even, so split pots never leave an odd
    chip — no odd-chip rule needed here.

Payoffs are net chip deltas per player and always sum to zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .cards import Card
from .evaluator import evaluate

# Streets
PREFLOP, FLOP, TURN, RIVER = 0, 1, 2, 3
STREET_NAMES = {PREFLOP: "preflop", FLOP: "flop", TURN: "turn", RIVER: "river"}

# Action kinds
FOLD, CHECK, CALL, RAISE = "fold", "check", "call", "raise"


@dataclass(frozen=True)
class Action:
    kind: str
    # For RAISE: the total chips this player will have committed on this street
    # AFTER the action (a "raise-to" amount). Ignored for fold/check/call.
    to_amount: int = 0

    def __str__(self) -> str:
        if self.kind == RAISE:
            return f"raise->{self.to_amount}"
        return self.kind


@dataclass
class HandState:
    # --- static for the hand ---
    sb: int
    bb: int
    button: int                        # player who posts SB / acts first preflop
    holes: list[list[Card]]            # holes[player] = [Card, Card]
    board: list[Card]                  # 5 community cards, known up front

    # --- dynamic ---
    stacks: list[int] = field(default_factory=lambda: [0, 0])          # chips behind
    street_committed: list[int] = field(default_factory=lambda: [0, 0])  # this street
    total_committed: list[int] = field(default_factory=lambda: [0, 0])   # this hand
    street: int = PREFLOP
    to_act: int = 0
    current_bet: int = 0               # highest street_committed this street
    last_raise_size: int = 0           # size of last full raise (for min-raise)
    has_acted: list[bool] = field(default_factory=lambda: [False, False])
    allin: list[bool] = field(default_factory=lambda: [False, False])
    facing_incomplete_allin: bool = False
    folded: int | None = None          # index of player who folded, if any
    finished: bool = False

    # ---- construction ----

    @classmethod
    def new_hand(cls, sb: int, bb: int, button: int,
                 stacks: list[int], holes: list[list[Card]],
                 board: list[Card]) -> "HandState":
        if len(board) != 5:
            raise ValueError("board must be exactly 5 cards (dealt up front)")
        s = cls(sb=sb, bb=bb, button=button,
                holes=[list(holes[0]), list(holes[1])], board=list(board),
                stacks=list(stacks))
        s._post_blinds()
        return s

    def _post_blinds(self) -> None:
        sb_player = self.button          # heads-up: button is the small blind
        bb_player = 1 - self.button
        self._commit(sb_player, min(self.sb, self.stacks[sb_player]))
        self._commit(bb_player, min(self.bb, self.stacks[bb_player]))
        self.current_bet = max(self.street_committed)
        self.last_raise_size = self.bb   # a full raise preflop is at least one BB
        # Blinds are forced, not voluntary action: has_acted stays False so the
        # big blind still gets the option after a limp.
        self.to_act = sb_player
        self._maybe_autofinish_preflop()

    def _commit(self, p: int, amount: int) -> None:
        """Move `amount` chips from stack p into the pot for this street."""
        amount = min(amount, self.stacks[p])
        self.stacks[p] -= amount
        self.street_committed[p] += amount
        self.total_committed[p] += amount
        if self.stacks[p] == 0 and amount > 0:
            self.allin[p] = True

    def _maybe_autofinish_preflop(self) -> None:
        # If a blind put a player all-in, there may be no more decisions to make
        # (e.g. both all-in from blinds). Let the normal advance logic settle it.
        if self.allin[0] and self.allin[1]:
            self._end_betting()

    # ---- queries ----

    def clone(self) -> "HandState":
        # Manual field copy — far cheaper than deepcopy on the CFR hot path.
        # holes/board are never mutated after construction, so they can be shared.
        s = HandState.__new__(HandState)
        s.sb = self.sb
        s.bb = self.bb
        s.button = self.button
        s.holes = self.holes
        s.board = self.board
        s.stacks = self.stacks[:]
        s.street_committed = self.street_committed[:]
        s.total_committed = self.total_committed[:]
        s.street = self.street
        s.to_act = self.to_act
        s.current_bet = self.current_bet
        s.last_raise_size = self.last_raise_size
        s.has_acted = self.has_acted[:]
        s.allin = self.allin[:]
        s.facing_incomplete_allin = self.facing_incomplete_allin
        s.folded = self.folded
        s.finished = self.finished
        return s

    def is_terminal(self) -> bool:
        return self.finished

    def to_call(self, p: int) -> int:
        return self.current_bet - self.street_committed[p]

    def _min_raise_to(self, p: int) -> int:
        return self.current_bet + self.last_raise_size

    def _max_raise_to(self, p: int) -> int:
        # all chips the player can put on this street
        return self.street_committed[p] + self.stacks[p]

    def legal_actions(self) -> list[Action]:
        """Legal actions for the player to act. RAISE is returned as a single
        canonical entry carrying (min_to, max_to) via two actions: the minimum
        legal raise and the all-in. Callers/bet abstractions can request any
        to_amount in [min_to, max_to]; use `raise_bounds` to size intermediate
        raises. We surface min and all-in so a naive caller always has valid
        moves.
        """
        if self.finished:
            return []
        p = self.to_act
        opp = 1 - p
        actions: list[Action] = []
        owed = self.to_call(p)

        if owed == 0:
            actions.append(Action(CHECK))
        else:
            actions.append(Action(FOLD))
            # A call never exceeds the player's stack (all-in call if short).
            actions.append(Action(CALL))

        # Can this player put in more chips (a bet or raise)?
        opp_can_contest = not self.allin[opp] and self.folded is None
        raise_allowed = (
            self.stacks[p] > owed          # has chips beyond just calling
            and opp_can_contest            # someone can respond
            and not self.facing_incomplete_allin  # under-raise all-in didn't reopen
        )
        if raise_allowed:
            lo, hi = self.raise_bounds(p)
            if lo is not None:
                actions.append(Action(RAISE, lo))
                if hi != lo:
                    actions.append(Action(RAISE, hi))  # all-in
        return actions

    def raise_bounds(self, p: int) -> tuple[int | None, int]:
        """(min_raise_to, max_raise_to) for player p, or (None, max) if p cannot
        raise. A short stack that can't afford a full min-raise may still shove
        (an incomplete raise); in that case min==max==all-in.
        """
        max_to = self._max_raise_to(p)
        if max_to <= self.current_bet:
            return None, max_to            # can't even exceed current bet
        min_to = self._min_raise_to(p)
        if min_to > max_to:
            # Can't make a full raise but can still go all-in for less.
            return max_to, max_to
        return min_to, max_to

    # ---- transitions ----

    def apply(self, action: Action) -> "HandState":
        s = self.clone()
        s._apply_inplace(action)
        return s

    def _apply_inplace(self, action: Action) -> None:
        if self.finished:
            raise ValueError("cannot act on a finished hand")
        p = self.to_act
        kind = action.kind

        if kind == FOLD:
            if self.to_call(p) == 0:
                raise ValueError("cannot fold when checking is free")
            self.folded = p
            self.finished = True
            return

        if kind == CHECK:
            if self.to_call(p) != 0:
                raise ValueError("cannot check facing a bet")
            self.has_acted[p] = True

        elif kind == CALL:
            owed = self.to_call(p)
            if owed <= 0:
                raise ValueError("nothing to call")
            self._commit(p, min(owed, self.stacks[p]))
            self.has_acted[p] = True

        elif kind == RAISE:
            self._apply_raise(p, action.to_amount)

        else:
            raise ValueError(f"unknown action kind: {kind!r}")

        self._advance(p)

    def _apply_raise(self, p: int, to_amount: int) -> None:
        lo, hi = self.raise_bounds(p)
        if lo is None:
            raise ValueError("raise not available")
        if not (lo <= to_amount <= hi):
            raise ValueError(f"raise-to {to_amount} outside legal [{lo}, {hi}]")
        if self.facing_incomplete_allin:
            raise ValueError("betting not reopened by the previous all-in")

        increment = to_amount - self.current_bet
        delta = to_amount - self.street_committed[p]
        self._commit(p, delta)

        if increment >= self.last_raise_size:
            # Full raise: update the min-raise yardstick and reopen action.
            self.last_raise_size = increment
            self.facing_incomplete_allin = False
        else:
            # Only reachable via an all-in short of a full raise.
            self.facing_incomplete_allin = True

        self.current_bet = to_amount
        self.has_acted[p] = True
        # Opponent must respond to the new bet.
        self.has_acted[1 - p] = False

    def _advance(self, actor: int) -> None:
        if self.finished:
            return
        opp = 1 - actor

        if self._opponent_must_act(opp):
            self.to_act = opp
            return

        # Betting round complete for this street.
        self._advance_street()

    def _opponent_must_act(self, opp: int) -> bool:
        if self.folded is not None:
            return False
        if self.allin[opp]:
            return False
        # Owes chips, or hasn't yet had a voluntary turn this street.
        return self.to_call(opp) > 0 or not self.has_acted[opp]

    def _advance_street(self) -> None:
        # If anyone is all-in, no further betting is possible; run to showdown.
        if self.allin[0] or self.allin[1] or self.street == RIVER:
            self._end_betting()
            return
        self.street += 1
        self.street_committed = [0, 0]
        self.current_bet = 0
        self.last_raise_size = self.bb
        self.has_acted = [False, False]
        self.facing_incomplete_allin = False
        # Postflop the non-button acts first.
        first = 1 - self.button
        self.to_act = first
        # If the first player can't act (all-in), the other can't contest alone.
        if self.allin[first]:
            self._end_betting()

    def _end_betting(self) -> None:
        self.finished = True

    # ---- settlement ----

    def payoffs(self) -> list[int]:
        """Net chip change per player for the hand (sums to zero). Returns the
        uncalled portion of any over-bet to its owner as part of the accounting.
        """
        if not self.finished:
            raise ValueError("hand not finished")

        tc0, tc1 = self.total_committed
        matched = min(tc0, tc1)
        # Only `matched` chips per player are truly at risk; any over-commit is
        # uncalled and returned, so it never enters this accounting. Net change
        # to a stack is therefore (-matched + chips won from the pot).
        result = [-matched, -matched]
        pot = 2 * matched

        if self.folded is not None:
            winner = 1 - self.folded
            result[winner] += pot
        else:
            w = self._showdown_winner()
            if w == -1:  # tie -> split (pot is even in heads-up)
                result[0] += pot // 2
                result[1] += pot // 2
            else:
                result[w] += pot
        return result

    def _showdown_winner(self) -> int:
        k0 = evaluate(self.holes[0] + self.board)
        k1 = evaluate(self.holes[1] + self.board)
        if k0 > k1:
            return 0
        if k1 > k0:
            return 1
        return -1
