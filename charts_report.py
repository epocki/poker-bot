"""Render solved preflop frequencies as colored 13x13 grids (RFI, 3-bet, ...).

Color scheme matches common solver UIs: red = raise/3-bet, green = call/limp,
blue = fold. Each cell's background blends those three by their frequency, and the
number printed is the headline frequency for that view (raise% for RFI, 3-bet%
for the defense view).
"""

from __future__ import annotations

import json
import sys

from engine.cards import RANK_TO_CHAR

RANKS_DESC = [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]
COMBOS = {"pair": 6, "s": 4, "o": 12}


def _class_at(rr: int, cr: int) -> str:
    r, c = RANK_TO_CHAR[rr], RANK_TO_CHAR[cr]
    if rr == cr:
        return r + c
    if rr > cr:
        return r + c + "s"
    return c + r + "o"


def _combos(cls: str) -> int:
    if len(cls) == 2:
        return COMBOS["pair"]
    return COMBOS[cls[2]]


def _mix(dist: dict) -> tuple[float, float, float]:
    """(raise, call, fold) frequencies from an action distribution."""
    if not dist:
        return (0.0, 0.0, 0.0)
    raise_ = sum(p for t, p in dist.items() if t.startswith("r"))
    call = dist.get("c", 0.0) + dist.get("x", 0.0)
    fold = dist.get("f", 0.0)
    return (raise_, call, fold)


def _cell(raise_: float, call: float, fold: float, headline: float) -> str:
    r, g, b = int(raise_ * 220) + 20, int(call * 200), int(fold * 200)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    fg = "30" if lum > 130 else "97"
    txt = f"{round(headline * 100):>3}"
    return f"\x1b[48;2;{r};{g};{b}m\x1b[{fg}m {txt} \x1b[0m"


def grid(avg: dict, history: str, headline: str = "raise") -> str:
    lines = ["     " + "".join(f"{RANK_TO_CHAR[r]:^5}" for r in RANKS_DESC)]
    for rr in RANKS_DESC:
        row = [f" {RANK_TO_CHAR[rr]} "]
        for cr in RANKS_DESC:
            cls = _class_at(rr, cr)
            d = avg.get(cls + ":" + history)
            raise_, call, fold = _mix(d) if d else (0.0, 0.0, 0.0)
            head = {"raise": raise_, "call": call, "fold": fold}[headline]
            row.append(_cell(raise_, call, fold, head))
        lines.append("".join(row))
    return "\n".join(lines)


def summary(avg: dict, history: str, label: str) -> str:
    """Combo-weighted aggregate frequencies across all 169 hands."""
    tot = ra = ca = fo = 0.0
    for rr in RANKS_DESC:
        for cr in RANKS_DESC:
            cls = _class_at(rr, cr)
            d = avg.get(cls + ":" + history)
            if not d:
                continue
            w = _combos(cls)
            r, c, f = _mix(d)
            ra += w * r
            ca += w * c
            fo += w * f
            tot += w
    if tot == 0:
        return f"{label}: (no data)"
    return (f"{label} (combo-weighted):  raise {ra/tot:5.1%}   "
            f"call {ca/tot:5.1%}   fold {fo/tot:5.1%}")


def open_size_mix(avg: dict) -> dict:
    """SB opening: combo-weighted frequency of each raise-size token."""
    acc: dict[str, float] = {}
    tot = 0.0
    for rr in RANKS_DESC:
        for cr in RANKS_DESC:
            cls = _class_at(rr, cr)
            d = avg.get(cls + ":")
            if not d:
                continue
            w = _combos(cls)
            tot += w
            for t, p in d.items():
                acc[t] = acc.get(t, 0.0) + w * p
    return {t: v / tot for t, v in acc.items()} if tot else {}


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "preflop_strategy.json"
    with open(path) as f:
        avg = json.load(f)

    # Does this solve ever open-limp? (any call/check mass at an opening node,
    # i.e. a key with empty history)
    limps = any(v.get("c", 0) + v.get("x", 0) > 1e-9
                for k, v in avg.items() if k.endswith(":"))
    green_label = "green=limp" if limps else "green=(none: no-limp solve)"

    print("=" * 70)
    print("SMALL BLIND / BUTTON — RFI (raise first in)   [100bb]")
    print(f"  number = raise%   red=raise  {green_label}  blue=fold")
    print("=" * 70)
    print(grid(avg, "", "raise"))
    print()
    print(" ", summary(avg, "", "SB open"))
    mix = open_size_mix(avg)
    sizes = "   ".join(f"{t}={p:.1%}" for t, p in sorted(mix.items(), key=lambda kv: -kv[1]))
    print("  SB open-size mix:", sizes)

    # Pick the modal raise open size for the defense chart.
    raise_sizes = {t: p for t, p in mix.items() if t.startswith("r")}
    modal = max(raise_sizes, key=raise_sizes.get)
    print()
    print("=" * 70)
    print(f"BIG BLIND — response facing SB open to {modal}")
    print("  number = 3-bet%   red=3bet  green=call/defend (not a limp)  blue=fold")
    print("=" * 70)
    print(grid(avg, modal, "raise"))
    print()
    print(" ", summary(avg, modal, f"BB vs {modal}"))
