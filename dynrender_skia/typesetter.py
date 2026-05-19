"""Professional CJK typesetting engine with kinsoku (禁則処理) line-breaking.

Implements algorithms from:
- W3C JLREQ (Japanese Layout Requirements)
- GB/T 15834-2011 (Chinese Punctuation Usage)
- Unicode UAX #14 (Line Breaking Algorithm)

Core rule: punctuation must never appear at the start of a line unless
it's opening punctuation, and opening punctuation must never appear at
the end of a line.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    import skia


class CharClass(Enum):
    """Typographic character classification for line-breaking decisions."""

    # CJK ideograph — break allowed before and after
    IDEOGRAPH = auto()
    # Latin letter or digit — break only at word/space boundaries
    ALPHABETIC = auto()
    # Space — mandatory break opportunity (space itself stays at line end)
    SPACE = auto()
    # Close/end punctuation: ) 】 」 』 》 〉 " ' etc. — break after, NOT before
    CLOSE = auto()
    # Open/start punctuation: ( 【 「 『 《 〈 " ' etc. — break before, NOT after
    OPEN = auto()
    # Non-starting punctuation: , . 、 。 ; : ? ! etc. — break after, NOT before
    NON_STARTER = auto()
    # Emoji character — break before and after
    EMOJI = auto()
    # Inline object (icon image) — break before and after
    INLINE_OBJECT = auto()
    # Explicit mandatory break (\\n)
    MANDATORY_BREAK = auto()
    # Middle dot, en-dash — can break before/after but prefers not to separate
    HYPHEN = auto()


# ---------------------------------------------------------------------------
# Unicode-based character classification
# ---------------------------------------------------------------------------

# Characters that MUST NOT appear at the start of a line
# CL (Close Punctuation) + NS (Non-Starter) per UAX #14 + GB/T 15834
_FORBIDDEN_LINE_START: set[str] = {
    # Close brackets — CJK and western
    ")", "】", "〕", "］", "」", "』", "》", "〉", "）",
    # Close quotes
    "\"", "'", "”", "’", "〃",
    # Periods, commas, ideographic comma/full-stop
    ".", ",", "。", "、", "，", "．",
    # Semicolons, colons
    ";", ":", "：", "；",
    # Question, exclamation
    "?", "!", "！", "？",
    # Other non-starters (ellipsis, percent, degree)
    "…", "％", "%", "℃",
    # Japanese kana small forms
    "ぁ", "ぃ", "ぅ", "ぇ", "ぉ",
    "っ", "ゃ", "ゅ", "ょ", "ゎ",
    "ァ", "ィ", "ゥ", "ェ", "ォ",
    "ッ", "ャ", "ュ", "ョ", "ヮ",
    "ㇰ", "ㇱ", "ㇲ", "ㇳ", "ㇴ",
    "ㇵ", "ㇶ",
}

# Characters that MUST NOT appear at the end of a line
# OP (Open Punctuation) per UAX #14
_FORBIDDEN_LINE_END: set[str] = {
    # Open brackets — CJK and western
    "(", "【", "〔", "［", "「", "『", "《", "〈", "（",
    # Open quotes
    "\"", "'", "“", "‘", "「", "『",
    # Currency symbols that precede numbers
    "$", "￥", "£", "¥",
}

# Characters that form inseparable pairs with following characters
_INSEPARABLE_BEFORE: set[str] = {"—", "―", "…", "……"}

# Characters that form inseparable pairs with preceding characters
_INSEPARABLE_AFTER: set[str] = {"—", "―"}

# CJK ideograph ranges (simplified)
_CJK_RANGES = [
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0x2F800, 0x2FA1F),  # CJK Compatibility Ideographs Supplement
    (0x3000, 0x303F),  # CJK Symbols and Punctuation (some)
    (0xFF00, 0xFFEF),  # Halfwidth and Fullwidth Forms
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0xAC00, 0xD7AF),  # Hangul Syllables
    (0x1100, 0x11FF),  # Hangul Jamo
]


def _is_cjk(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def classify_char(ch: str) -> CharClass:
    """Classify a single character for line-breaking purposes."""
    if ch in _FORBIDDEN_LINE_START:
        return CharClass.NON_STARTER if ch in ",.，。、;；:：?!！？、。，．：；！？…%％" else CharClass.CLOSE
    if ch in _FORBIDDEN_LINE_END:
        return CharClass.OPEN
    cp = ord(ch)
    if _is_cjk(cp):
        return CharClass.IDEOGRAPH
    if ch.isspace():
        return CharClass.SPACE
    if ch in ("-", "‐", "‑", "‒", "–", "—", "―"):
        return CharClass.HYPHEN
    if ch in _INSEPARABLE_BEFORE:
        return CharClass.NON_STARTER
    if ch in _INSEPARABLE_AFTER:
        return CharClass.OPEN
    return CharClass.ALPHABETIC


# ---------------------------------------------------------------------------
# Typographic atom — the unit of typesetting
# ---------------------------------------------------------------------------


@dataclass
class Atom:
    """A single typesetting unit — a character or inline object with its measured width."""

    text: str
    width: float
    char_class: CharClass
    font: Optional[skia.Font] = None
    # For richtext: the paint color for this atom
    paint_color: Optional[skia.Color] = None
    # For inline icon objects: the icon image
    icon_image: Optional[skia.Image] = None


# ---------------------------------------------------------------------------
# Kinsoku line breaker
# ---------------------------------------------------------------------------


@dataclass
class _LineCandidate:
    atoms: list[Atom] = field(default_factory=list)
    total_width: float = 0.0


class KinsokuLineBreaker:
    """Greedy line breaker with kinsoku correction passes.

    Usage::

        breaker = KinsokuLineBreaker(max_width=900, indent=40)
        atoms = breaker.atomize(text, measure_fn, emoji_map)
        lines = breaker.break_lines(atoms)

    ``lines`` is a list of (start_idx, end_idx) tuples into the atoms list.
    """

    def __init__(self, max_width: float, indent: float = 0.0) -> None:
        self.max_width = max_width
        self.indent = indent
        self._first_line = True

    def break_lines(self, atoms: list[Atom]) -> list[tuple[int, int]]:
        """Partition *atoms* into lines, returning (start, end) index pairs.

        The end index is exclusive, following Python slicing convention.
        """
        if not atoms:
            return []

        lines: list[tuple[int, int]] = []
        i = 0
        n = len(atoms)
        line_width = self.max_width if self._first_line else self.max_width
        self._first_line = False

        while i < n:
            j = self._fill_line(atoms, i, n, line_width)
            j = self._apply_kinsoku_pullback(atoms, i, j, n)
            j = self._apply_kinsoku_pushforward(atoms, i, j, n)
            # Safety: if no progress, force at least one atom
            if j <= i:
                j = i + 1
            lines.append((i, j))
            i = j
            line_width = self.max_width  # subsequent lines use full width

        return lines

    def _fill_line(self, atoms: list[Atom], start: int, n: int, max_w: float) -> int:
        """Greedy fill: return the first index *after* the filled atoms."""
        width = 0.0
        i = start
        while i < n:
            atom = atoms[i]
            if atom.char_class == CharClass.MANDATORY_BREAK:
                i += 1  # skip the break character
                break
            if width + atom.width > max_w and width > 0:
                break
            width += atom.width
            i += 1
        return i

    def _apply_kinsoku_pullback(self, atoms: list[Atom], line_start: int, break_at: int, n: int) -> int:
        """Pull back forbidden line-start characters to the previous line.

        If *break_at* is the first char of the next line and it's forbidden
        at line start, move the last char(s) of the current line down.
        Continue until the first char of the next line is allowed at line start.
        """
        while break_at < n and break_at > line_start:
            first_of_next = atoms[break_at]
            if not self._is_forbidden_at_line_start(first_of_next):
                break
            # Pull the last character from the current line into the next line
            break_at -= 1
            # If we pulled a close/non-starter before it, keep pulling
            while break_at > line_start and self._is_forbidden_at_line_end(atoms[break_at - 1]):
                break_at -= 1
        return break_at

    def _apply_kinsoku_pushforward(self, atoms: list[Atom], line_start: int, break_at: int, n: int) -> int:
        """Push forbidden line-end characters to the next line.

        If the last char of the current line is forbidden at line end,
        move it to the next line.
        """
        while break_at > line_start and break_at <= n:
            last_of_line = atoms[break_at - 1]
            if not self._is_forbidden_at_line_end(last_of_line):
                break
            break_at -= 1
        return break_at

    @staticmethod
    def _is_forbidden_at_line_start(atom: Atom) -> bool:
        return atom.char_class in (CharClass.CLOSE, CharClass.NON_STARTER)

    @staticmethod
    def _is_forbidden_at_line_end(atom: Atom) -> bool:
        return atom.char_class == CharClass.OPEN


# ---------------------------------------------------------------------------
# Convenience: build atoms from text
# ---------------------------------------------------------------------------


def atomize_text(
    text: str,
    measure_fn,
    emoji_map: dict[int, tuple[int, str]],
    default_font: skia.Font,
    emoji_font: skia.Font,
) -> list[Atom]:
    """Convert text to a list of Atoms, including emoji resolution.

    Args:
        text: The input text.
        measure_fn: Callable (char, font) -> float measuring the rendered width.
        emoji_map: Dict of {start_offset: (end_offset, emoji_char)} from emoji.emoji_list.
        default_font: Font for non-emoji characters.
        emoji_font: Font for emoji characters.

    Returns:
        List of Atom objects ready for line-breaking.
    """
    atoms: list[Atom] = []
    total = len(text)
    offset = 0

    while offset < total:
        ch = text[offset]
        if ch == "\n":
            atoms.append(Atom("\n", 0.0, CharClass.MANDATORY_BREAK))
            offset += 1
            continue

        if offset in emoji_map:
            end_pos, emoji_char = emoji_map[offset]
            offset = end_pos
            font = emoji_font
            ch = emoji_char
        else:
            offset += 1
            font = default_font

        w = measure_fn(ch, font)
        atom_class = classify_char(ch)
        # Override class for emoji
        if atom_class in (CharClass.CLOSE, CharClass.OPEN, CharClass.NON_STARTER):
            # Emoji char that happens to look like punctuation — trust the original
            pass
        atoms.append(Atom(ch, w, atom_class, font))

    return atoms
