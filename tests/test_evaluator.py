"""Evaluator correctness — categories, tiebreakers, and the nasty edge cases."""

import random

from engine.cards import make_deck, parse_cards
from engine.evaluator import (
    evaluate, evaluate7_fast, score5, category_of,
    HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH,
    FULL_HOUSE, QUADS, STRAIGHT_FLUSH,
)


def ev(s):
    return evaluate(parse_cards(s))


def cat(s):
    return category_of(ev(s))


# --- category detection on exactly 5 cards ---

def test_categories_5card():
    assert cat("As Ks Qs Js Ts") == STRAIGHT_FLUSH
    assert cat("9h 9c 9d 9s 2c") == QUADS
    assert cat("8h 8c 8d 2s 2c") == FULL_HOUSE
    assert cat("Ah Kh 9h 4h 2h") == FLUSH
    assert cat("5h 4c 3d 2s Ac") == STRAIGHT   # wheel
    assert cat("7h 8c 9d Ts Jc") == STRAIGHT
    assert cat("Qh Qc Qd 4s 2c") == TRIPS
    assert cat("Kh Kc 3d 3s 2c") == TWO_PAIR
    assert cat("Jh Jc 9d 5s 2c") == PAIR
    assert cat("Ah Kc 9d 5s 2c") == HIGH_CARD


# --- the wheel: A-2-3-4-5 is the lowest straight, high card = 5 ---

def test_wheel_is_five_high_straight():
    wheel = ev("5h 4c 3d 2s Ac")
    six_high = ev("6h 5c 4d 3s 2c")
    assert category_of(wheel) == STRAIGHT
    assert wheel[1] == 5            # five-high, not ace-high
    assert six_high > wheel         # 6-high straight beats the wheel


def test_wheel_straight_flush():
    steel_wheel = ev("5s 4s 3s 2s As")
    six_high_sf = ev("6s 5s 4s 3s 2s")
    assert category_of(steel_wheel) == STRAIGHT_FLUSH
    assert steel_wheel[1] == 5
    assert six_high_sf > steel_wheel


def test_broadway_beats_wheel():
    assert ev("As Ks Qs Js Ts") > ev("5s 4s 3s 2s As")


# --- ace-high vs king-high flush kicker ordering ---

def test_flush_kickers():
    assert ev("Ah Kh 9h 4h 2h") > ev("Kh Qh 9h 4h 2h")
    # same top four, last kicker decides
    assert ev("Ah Kh 9h 4h 3h") > ev("Ah Kh 9h 4h 2h")


# --- full house: trips rank dominates the pair rank ---

def test_full_house_ordering():
    # trips rank leads the tiebreak: KKK22 beats 222AA despite the lower pair
    assert ev("Kh Kc Kd 2s 2c") > ev("2h 2c 2d Ah Ac")


def test_full_house_pair_tiebreak():
    assert ev("9h 9c 9d Ah Ac") > ev("9h 9c 9d Kh Kc")


# --- two pair tiebreak: high pair, then low pair, then kicker ---

def test_two_pair_tiebreak():
    assert ev("Ah Ac 2d 2s Kc") > ev("Kh Kc Qd Qs Ac")   # aces up beats kings up
    assert ev("Ah Ac Kd Ks 5c") > ev("Ah Ac Kd Ks 4c")   # kicker


# --- quads kicker ---

def test_quads_kicker():
    assert ev("9h 9c 9d 9s Ac") > ev("9h 9c 9d 9s Kc")


# --- best-5-of-7 selection ---

def test_seven_card_picks_best_five():
    # board makes a straight; hole cards are irrelevant junk
    key = ev("2c 7d 9h Ts Jc Qd Kh")   # T J Q K + 9 -> 9-K straight; also T-K? need A. best = 9TJQK
    assert category_of(key) == STRAIGHT
    assert key[1] == 13   # king-high straight (9,T,J,Q,K)


def test_seven_card_flush_over_straight():
    # 5 hearts present -> flush must be chosen even though a straight also exists
    key = ev("6h 7h 8h 9h Th 2c 3d")
    assert category_of(key) == STRAIGHT_FLUSH  # actually 6-T all hearts = straight flush


def test_seven_card_full_house_from_two_pair_plus_trip():
    key = ev("Kh Kc Kd 5s 5h 2c 7d")
    assert category_of(key) == FULL_HOUSE
    assert key[1] == 13 and key[2] == 5


def test_two_pair_from_seven_uses_best_two():
    # three pairs present; must use the top two pairs + best kicker
    key = ev("Ah Ac Kh Kc Qh Qc 2d")
    assert category_of(key) == TWO_PAIR
    assert key[1] == 14 and key[2] == 13 and key[3] == 12  # AA KK, Q kicker


# --- exact ties ---

def test_identical_hands_tie():
    a = ev("Ah Kh Qc Jd Ts 3c 2d")
    b = ev("As Ks Qh Jc Th 3d 2s")
    assert a == b


# --- fast 7-card evaluator must agree with the reference on random hands ---

def test_fast_evaluator_matches_reference():
    rng = random.Random(1234)
    deck = make_deck()
    for _ in range(20000):
        rng.shuffle(deck)
        hand = deck[:7]
        assert evaluate(hand) == evaluate7_fast(hand)
