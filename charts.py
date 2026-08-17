"""Render preflop strategies as the familiar 13x13 starting-hand grid.

Rows/cols run A..2. Upper triangle = suited, lower triangle = offsuit, diagonal =
pairs (matching how solvers/charts display ranges).
"""

from __future__ import annotations

from engine.cards import RANK_TO_CHAR
from engine.state import FOLD, CHECK, CALL

RANKS_DESC = [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]


def _class_at(row_rank: int, col_rank: int) -> str:
    r, c = RANK_TO_CHAR[row_rank], RANK_TO_CHAR[col_rank]
    if row_rank == col_rank:
        return r + c            # pair
    if row_rank > col_rank:
        return r + c + "s"      # upper triangle: suited
    return c + r + "o"          # lower triangle: offsuit


def _bucket(dist: dict[str, float]) -> str:
    """One-character summary of a decision's mix.
      . fold-dominant   L limp/check   R raise (non all-in)   S shove (all-in)
    Chooses by which *category* holds the most probability."""
    if not dist:
        return " "
    fold = dist.get("f", 0.0)
    limp = dist.get("x", 0.0) + dist.get("c", 0.0)
    shove = 0.0
    raise_ = 0.0
    for t, p in dist.items():
        if t.startswith("r"):
            # the all-in token is the largest raise; treat it separately
            raise_ += p
    # distinguish shove: the raise token equal to the all-in amount is handled by
    # the caller passing an all-in token set; here approximate by largest raise.
    cats = {".": fold, "L": limp, "R": raise_, "S": shove}
    return max(cats, key=cats.get)


def open_grid(avg: dict, allin_token: str | None = None) -> str:
    """ASCII grid of the small blind's opening decision (empty history)."""
    # Reclassify shove vs raise using the actual all-in token if provided.
    def bucket(cls: str) -> str:
        d = avg.get(cls + ":")
        if not d:
            return " "
        fold = d.get("f", 0.0)
        limp = d.get("x", 0.0) + d.get("c", 0.0)
        shove = d.get(allin_token, 0.0) if allin_token else 0.0
        raise_ = sum(p for t, p in d.items()
                     if t.startswith("r") and t != allin_token)
        cats = {".": fold, "L": limp, "R": raise_, "S": shove}
        return max(cats, key=cats.get)

    lines = ["    " + " ".join(RANK_TO_CHAR[r] for r in RANKS_DESC)]
    for rr in RANKS_DESC:
        row = [RANK_TO_CHAR[rr] + " "]
        for cr in RANKS_DESC:
            row.append(bucket(_class_at(rr, cr)))
        lines.append(" " + " ".join(row))
    lines.append("")
    lines.append("  legend:  . fold   L limp/call   R raise   S shove(all-in)")
    return "\n".join(lines)


def describe(avg: dict, cls: str, history: str = "") -> str:
    d = avg.get(cls + ":" + history)
    if not d:
        return f"{cls} [{history or 'open'}]: (unreached)"
    parts = [f"{t}={p:.0%}" for t, p in sorted(d.items(), key=lambda kv: -kv[1]) if p > 0.005]
    return f"{cls} [{history or 'open'}]: " + "  ".join(parts)
