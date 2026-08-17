# Heads-Up NLHE Preflop CFR+ Solver

A from-scratch heads-up No-Limit Hold'em engine and preflop strategy solver built in Python. The project implements poker rules and hand evaluation, constructs an abstracted preflop game tree, and trains approximate equilibrium strategies using Counterfactual Regret Minimization (CFR+).

The solver includes both a Monte Carlo CFR implementation and a vectorized full-width CFR+ implementation that operates over all 169 canonical starting-hand classes at once.

## What it includes

- **Heads-up NLHE betting engine** with blind posting, preflop/postflop action order, minimum raises, all-ins, incomplete raises, uncalled-bet returns, and zero-sum payoff accounting.
- **Poker hand evaluator** for 5-7 card hands, including a faster direct 7-card evaluator used in simulation-heavy code.
- **169-class preflop equity model** for hands such as `AA`, `AKs`, and `AKo`, with card removal respected between players.
- **CFR+ solver** using sampled deals and regret matching.
- **Vectorized CFR+ solver** using NumPy to update all hand classes in a single tree traversal.
- **Exact best-response exploitability calculation** reported in big blinds per 100 hands (`bb/100`).
- **Preflop strategy visualization** as 13x13 starting-hand grids and action-frequency reports.
- **Self-play harness** for running complete heads-up hands between baseline agents.
- **Automated tests** for hand evaluation and betting-state edge cases.

## Project structure

```text
poker-bot/
├── engine/
│   ├── cards.py          # Card/deck primitives
│   ├── dealer.py         # Heads-up dealing
│   ├── evaluator.py      # 5-7 card hand evaluation
│   ├── equity.py         # 169x169 preflop equity table
│   └── state.py          # NLHE betting state machine
├── agents.py             # Baseline agents
├── preflop_game.py       # Preflop game abstraction for CFR
├── tree.py               # Static betting-tree construction
├── cfr.py                # Sampled CFR+ trainer
├── cfr_vec.py            # Vectorized full-width CFR+ trainer
├── exploitability.py     # Exact best-response / NashConv metric
├── selfplay.py           # End-to-end self-play simulation
├── charts.py             # Text strategy grids
├── charts_report.py      # Colored frequency reports
├── tests/                # Pytest test suite
├── preflop_strategy.json
└── preflop_strategy_nolimp.json
```

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pytest
```

The repository includes a cached preflop equity table, so you can use the solver without rebuilding equities from scratch.

## Run the tests

```bash
pytest -q
```

The current test suite covers poker-hand ranking and key heads-up betting rules, including all-ins, minimum raises, incomplete raises, betting order, ties, and chip conservation.

## Run a self-play simulation

```bash
python selfplay.py --hands 5000 --seed 42
```

This plays hands between a random-action agent and a calling-station baseline while checking that the engine conserves chips.

## Train a preflop strategy

The vectorized solver is the faster implementation:

```python
import json

from engine.equity import EquityTable
from preflop_game import PreflopGame
from cfr_vec import VectorTrainer

# Load cached 169x169 preflop equities.
eq = EquityTable.load()

# 100bb heads-up game: blinds 1/2, 200-chip stacks.
game = PreflopGame(
    eq,
    sb=1,
    bb=2,
    stack=200,
    raise_fractions=(0.5, 1.0),
    no_open_limp=True,
)

trainer = VectorTrainer(game)
trainer.train(iterations=2000, log_every=100)

strategy = trainer.average_strategy()
with open("strategy.json", "w") as f:
    json.dump(strategy, f)
```

The action space uses an abstraction rather than every possible no-limit bet size: half-pot, pot, and all-in raises are included where legal.

## Measure exploitability

A strategy can be evaluated against an exact best response:

```python
from exploitability import exploitability_bb100

print(exploitability_bb100(trainer, bb=2))
```

Lower exploitability indicates a strategy closer to equilibrium within the modeled preflop game.

## Visualize a solved strategy

The repository includes solved strategy JSON files that can be rendered as 13x13 preflop grids:

```bash
python charts_report.py preflop_strategy_nolimp.json
```

The report shows action frequencies for each starting-hand class and aggregate frequencies across all combinations.

## Model scope

This is currently a preflop solver, not a complete postflop NLHE solver. Preflop betting is modeled explicitly, while terminal showdowns are valued using precomputed all-in equity. The betting tree also uses a reduced raise-size abstraction to keep CFR training tractable.

## Notes on implementation

The vectorized solver maintains per-hand-class regrets and average-strategy weights at each decision node. Card-removal effects are preserved through a joint deal-weight matrix, and exploitability is computed using an exact best response over that same weighted hand distribution.
