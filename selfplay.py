"""Self-play harness: runs N heads-up hands between two agents and reports the
net chip flow. Verifies the whole engine loop (deal -> bet -> settle) end-to-end
and that chips are conserved.
"""

from __future__ import annotations

import argparse

from engine.dealer import Dealer
from engine.state import HandState
from agents import RandomAgent, CallingStation


def play_hand(agents, sb, bb, button, stacks, dealer) -> list[int]:
    holes, board = dealer.deal()
    s = HandState.new_hand(sb, bb, button, stacks, holes, board)
    guard = 0
    while not s.is_terminal():
        agent = agents[s.to_act]
        s = s.apply(agent.act(s))
        guard += 1
        if guard > 200:
            raise RuntimeError("hand did not terminate")
    return s.payoffs()


def run(n_hands: int, seed: int) -> None:
    dealer = Dealer(seed)
    agents = [RandomAgent(seed + 1), CallingStation()]
    sb, bb, start = 1, 2, 200
    net = [0, 0]
    button = 0
    for _ in range(n_hands):
        # both start each hand with the reference stack (cash game: top up)
        pay = play_hand(agents, sb, bb, button, [start, start], dealer)
        net[0] += pay[0]
        net[1] += pay[1]
        assert pay[0] + pay[1] == 0, "chips not conserved!"
        button = 1 - button
    bb_per_100 = (net[0] / bb) / n_hands * 100
    print(f"hands: {n_hands}")
    print(f"P0 (RandomAgent) net: {net[0]:+d} chips  ({bb_per_100:+.1f} bb/100)")
    print(f"P1 (CallingStation) net: {net[1]:+d} chips")
    print(f"chips conserved: {net[0] + net[1] == 0}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--hands", type=int, default=5000)
    ap.add_argument("-s", "--seed", type=int, default=42)
    args = ap.parse_args()
    run(args.hands, args.seed)
