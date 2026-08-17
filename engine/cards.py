"""Card, rank, suit, and deck primitives.

Ranks are ints 2..14 (14 = Ace). Suits are single chars 'c','d','h','s'.
A Card is a frozen (rank, suit) with a compact 2-char string form, e.g. 'As', 'Td', '2c'.
"""

from __future__ import annotations

from dataclasses import dataclass

RANKS = list(range(2, 15))  # 2..14, where 14 is Ace
SUITS = ("c", "d", "h", "s")

# Human-facing rank glyphs. 10 is 'T'.
RANK_TO_CHAR = {14: "A", 13: "K", 12: "Q", 11: "J", 10: "T",
                9: "9", 8: "8", 7: "7", 6: "6", 5: "5", 4: "4", 3: "3", 2: "2"}
CHAR_TO_RANK = {v: k for k, v in RANK_TO_CHAR.items()}


@dataclass(frozen=True, order=True)
class Card:
    rank: int
    suit: str

    def __post_init__(self) -> None:
        if self.rank not in RANK_TO_CHAR:
            raise ValueError(f"invalid rank: {self.rank!r}")
        if self.suit not in SUITS:
            raise ValueError(f"invalid suit: {self.suit!r}")

    def __str__(self) -> str:
        return f"{RANK_TO_CHAR[self.rank]}{self.suit}"

    def __repr__(self) -> str:
        return f"Card('{self}')"

    @classmethod
    def parse(cls, s: str) -> "Card":
        """Parse a 2-char string like 'As' or 'Td' into a Card."""
        s = s.strip()
        if len(s) != 2:
            raise ValueError(f"card string must be 2 chars, got {s!r}")
        rank_char, suit = s[0].upper(), s[1].lower()
        if rank_char not in CHAR_TO_RANK:
            raise ValueError(f"invalid rank char: {rank_char!r}")
        if suit not in SUITS:
            raise ValueError(f"invalid suit char: {suit!r}")
        return cls(CHAR_TO_RANK[rank_char], suit)


def make_deck() -> list[Card]:
    """Return a fresh ordered 52-card deck."""
    return [Card(r, s) for r in RANKS for s in SUITS]


def parse_cards(s: str) -> list[Card]:
    """Parse a space- or comma-separated string of cards, e.g. 'As Kd 2c'."""
    tokens = s.replace(",", " ").split()
    return [Card.parse(t) for t in tokens]
