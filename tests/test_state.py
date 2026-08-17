"""Heads-up NLHE betting-engine correctness, focused on cash-game edge cases."""

import pytest

from engine.cards import parse_cards
from engine.state import (
    HandState, Action, FOLD, CHECK, CALL, RAISE,
    PREFLOP, FLOP, TURN, RIVER,
)

# A board where holes decide the winner unless noted.
BOARD = parse_cards("2c 7d 9h Jc Qs")
P0_WINS = [parse_cards("Ah Ad"), parse_cards("Ks Kd")]   # aces beat kings
TIE_BOARD = parse_cards("As Ks Qs Js Ts")                 # royal on board -> tie


def new(stacks, button=0, holes=None, board=BOARD, sb=1, bb=2):
    holes = holes or [parse_cards("Ah Ad"), parse_cards("Ks Kd")]
    return HandState.new_hand(sb, bb, button, stacks, holes, board)


def kinds(state):
    return {a.kind for a in state.legal_actions()}


# --- blind posting & preflop order ---

def test_blinds_and_preflop_order():
    s = new([100, 100], button=0)
    assert s.total_committed == [1, 2]   # button posts SB, other posts BB
    assert s.current_bet == 2
    assert s.to_act == 0                 # button (SB) acts first preflop
    assert kinds(s) == {FOLD, CALL, RAISE}


def test_min_open_is_two_bb():
    s = new([100, 100], button=0)
    lo, hi = s.raise_bounds(0)
    assert lo == 4 and hi == 100        # min open = raise-to 2*BB; max = shove


# --- limp + big-blind option ---

def test_limp_gives_bb_option_then_flop():
    s = new([100, 100], button=0)
    s = s.apply(Action(CALL))            # button limps to 2
    assert s.street == PREFLOP           # not over: BB has the option
    assert s.to_act == 1
    assert kinds(s) == {CHECK, RAISE}    # BB can check or raise, not fold-for-free
    s = s.apply(Action(CHECK))           # BB checks option
    assert s.street == FLOP
    assert s.to_act == 1                 # postflop the non-button acts first


# --- postflop acting order ---

def test_postflop_non_button_first():
    s = new([100, 100], button=0)
    s = s.apply(Action(CALL)).apply(Action(CHECK))   # to flop
    assert s.street == FLOP
    assert s.to_act == 1
    s = s.apply(Action(CHECK))
    assert s.to_act == 0                 # button acts last postflop


# --- fold returns the uncalled bet ---

def test_fold_returns_uncalled_bet():
    s = new([100, 100], button=0)
    s = s.apply(Action(RAISE, 10))       # button raises to 10
    s = s.apply(Action(FOLD))            # BB folds
    assert s.is_terminal()
    # Only the BB's 2 was matched; button's extra 8 is uncalled and returned.
    assert s.payoffs() == [2, -2]
    assert sum(s.payoffs()) == 0


def test_cannot_fold_when_check_is_free():
    s = new([100, 100], button=0)
    s = s.apply(Action(CALL))            # limp; BB to act, owes 0
    with pytest.raises(ValueError):
        s.apply(Action(FOLD))


# --- all-in under-raise does NOT reopen betting ---

def test_incomplete_allin_does_not_reopen():
    # p1 (BB) is short and can only shove for less than a full re-raise.
    s = new([100, 8], button=0)
    s = s.apply(Action(RAISE, 6))        # button raises to 6 (full raise, +4)
    assert s.to_act == 1
    lo, hi = s.raise_bounds(1)
    assert lo == 8 and hi == 8           # can only shove to 8 (incomplete raise)
    s = s.apply(Action(RAISE, 8))        # BB shoves incomplete
    assert s.facing_incomplete_allin
    assert s.to_act == 0
    # Button already acted; the short shove did NOT reopen -> call or fold only.
    assert kinds(s) == {FOLD, CALL}


def test_full_reraise_does_reopen():
    s = new([100, 100], button=0)
    s = s.apply(Action(RAISE, 6))        # button to 6 (+4)
    s = s.apply(Action(RAISE, 12))       # BB reraises to 12 (+6 >= 4, full)
    assert not s.facing_incomplete_allin
    assert RAISE in kinds(s)             # button may reraise again
    lo, _ = s.raise_bounds(0)
    assert lo == 18                      # min reraise-to = 12 + last raise (6)


# --- all-in showdown with unequal stacks: excess returned, matched pot ---

def test_allin_showdown_unequal_stacks():
    # Button 50, BB 30. Button shoves, BB calls all-in for 30.
    s = new([50, 30], button=0, holes=P0_WINS)
    s = s.apply(Action(RAISE, 50))       # button shoves to 50
    s = s.apply(Action(CALL))            # BB calls all-in (only 30)
    assert s.is_terminal()
    # Matched = 30 each; button's extra 20 returned. Button wins the 60 pot.
    assert s.payoffs() == [30, -30]
    assert sum(s.payoffs()) == 0


def test_allin_runs_out_all_streets():
    # If all-in happens preflop, remaining board is already known; goes to showdown.
    s = new([100, 100], button=0, holes=P0_WINS)
    s = s.apply(Action(RAISE, 100))      # button shoves
    s = s.apply(Action(CALL))            # BB calls all-in
    assert s.is_terminal()
    assert s.payoffs() == [100, -100]


# --- tie splits evenly (no odd chip possible heads-up) ---

def test_tie_splits_pot():
    s = new([100, 100], button=0, holes=[parse_cards("2h 3h"), parse_cards("4d 5d")],
            board=TIE_BOARD)
    s = s.apply(Action(CALL)).apply(Action(CHECK))          # limped flop
    # check down all streets
    while not s.is_terminal():
        s = s.apply(Action(CHECK))
    assert s.payoffs() == [0, 0]         # both play the board


# --- payoffs always sum to zero across a fuzz of lines ---

def test_payoffs_zero_sum_fuzz():
    import itertools
    lines = [
        [Action(FOLD)],
        [Action(RAISE, 6), Action(FOLD)],
        [Action(CALL), Action(RAISE, 8), Action(CALL)],
        [Action(RAISE, 100), Action(CALL)],
        [Action(RAISE, 6), Action(RAISE, 18), Action(CALL)],
    ]
    for button in (0, 1):
        for line in lines:
            s = new([100, 100], button=button, holes=P0_WINS)
            for a in line:
                if s.is_terminal():
                    break
                s = s.apply(a)
            # play any remainder out by checking/calling to reach terminal
            guard = 0
            while not s.is_terminal() and guard < 20:
                la = kinds(s)
                a = Action(CHECK) if CHECK in la else Action(CALL)
                s = s.apply(a)
                guard += 1
            assert s.is_terminal()
            assert sum(s.payoffs()) == 0
