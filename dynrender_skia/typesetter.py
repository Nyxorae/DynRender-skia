"""Professional CJK typesetting engine with kinsoku (禁則処理) line-breaking.

Implements algorithms from:
- W3C JLREQ (Japanese Layout Requirements)
- GB/T 15834-2011 (Chinese Punctuation Usage)
- Unicode UAX #14 (Line Breaking Algorithm)
- Knuth-Plass optimal line breaking (TeX's algorithm)

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
    """Classify a character (or compound emoji) for line-breaking purposes.

    Multi-character strings (emoji + variation selector) are classified
    by their first code-point only.
    """
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
    """Greedy line breaker with two-pass kinsoku correction.

    **Algorithm** (per line):

    1. **Greedy fill** — accumulate atoms until ``max_width`` is exceeded,
       recording the overflow point as a candidate break.
    2. **Pull-back (追い出し)** — if the first character of the *next*
       line is forbidden at line-start (e.g. ``。、，》〕」…``), pull the
       last character(s) of the *current* line down.  Repeat until the
       next line's first character is allowed.
    3. **Push-forward (追い込み)** — if the last character of the
       *current* line is forbidden at line-end (e.g. ``《「『（$``),
       push it to the next line.

    Usage::

        breaker = KinsokuLineBreaker(max_width=900, indent=40)
        lines = breaker.break_lines(atoms)
        for start, end in lines:
            draw_line(atoms[start:end])

    ``lines`` is a list of ``(start, end)`` index pairs (end exclusive).
    """

    def __init__(self, max_width: float, indent: float = 0.0) -> None:
        self.max_width = max_width
        self.indent = indent
        self._first_line = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def break_lines(self, atoms: list[Atom]) -> list[tuple[int, int]]:
        """Partition *atoms* into kinsoku-corrected lines.

        Returns:
            List of ``(start, end)`` index pairs (end exclusive,
            following Python slicing convention).
        """
        if not atoms:
            return []

        lines: list[tuple[int, int]] = []
        i = 0
        n = len(atoms)
        line_width = self.max_width  # first line honours indent in future
        self._first_line = False

        while i < n:
            # ---- 1. greedy fill ----
            j = self._fill_line(atoms, i, n, line_width)

            # ---- 2. pull-back: fix forbidden line-start ----
            j = self._apply_kinsoku_pullback(atoms, i, j, n)

            # ---- 3. push-forward: fix forbidden line-end ----
            j = self._apply_kinsoku_pushforward(atoms, i, j, n)

            # Safety valve: if corrections consumed everything, force 1 atom
            if j <= i:
                j = i + 1

            lines.append((i, j))
            i = j
            line_width = self.max_width

        return lines

    # ------------------------------------------------------------------
    # Pass 1 — greedy fill
    # ------------------------------------------------------------------

    def _fill_line(
        self, atoms: list[Atom], start: int, n: int, max_w: float
    ) -> int:
        """Return the index just *after* the last atom that fits in *max_w*.

        Stops at MANDATORY_BREAK (``\\n``) — the break character is consumed
        but not included in the line.
        """
        width = 0.0
        i = start
        while i < n:
            atom = atoms[i]
            if atom.char_class == CharClass.MANDATORY_BREAK:
                i += 1       # consume the newline
                break
            if width + atom.width > max_w and width > 0:
                break        # overflow — break before this atom
            width += atom.width
            i += 1
        return i

    # ------------------------------------------------------------------
    # Pass 2 — kinsoku pull-back (追い出し)
    # ------------------------------------------------------------------

    def _apply_kinsoku_pullback(
        self, atoms: list[Atom], line_start: int, break_at: int, n: int
    ) -> int:
        """Move forbidden line-start characters to the *previous* line.

        When the atom at *break_at* (first of the next line) has type
        ``CLOSE`` or ``NON_STARTER``, this method walks backward,
        pulling the offending character(s) from the current line end
        into the next line's beginning.

        Example::

            今天天气真好，       ← break_at would be after "，"
            我想出门。           ← but "，" is forbidden at line start!

        After pull-back::

            今天天气真好         ← "，" pulled to next line
            ，我想出门。
        """
        while break_at < n and break_at > line_start:
            first_of_next = atoms[break_at]
            if not self._is_forbidden_at_line_start(first_of_next):
                break
            break_at -= 1
            # Continue pulling if the preceding char is forbidden at line-end.
            # Handles chains like  "《text》"  → 《 must not end a line.
            while break_at > line_start and self._is_forbidden_at_line_end(
                atoms[break_at - 1]
            ):
                break_at -= 1
        return break_at

    # ------------------------------------------------------------------
    # Pass 3 — kinsoku push-forward (追い込み)
    # ------------------------------------------------------------------

    def _apply_kinsoku_pushforward(
        self, atoms: list[Atom], line_start: int, break_at: int, n: int
    ) -> int:
        """Push forbidden line-end characters to the *next* line.

        When the atom at *break_at - 1* (last of the current line) has type
        ``OPEN``, it is moved to the following line.
        """
        while break_at > line_start and break_at <= n:
            last_of_line = atoms[break_at - 1]
            if not self._is_forbidden_at_line_end(last_of_line):
                break
            break_at -= 1
        return break_at

    # ------------------------------------------------------------------
    # Kinsoku predicates
    # ------------------------------------------------------------------

    @staticmethod
    def _is_forbidden_at_line_start(atom: Atom) -> bool:
        """Return True when *atom* must NOT appear at the start of a line.

        This covers:
        - ``CLOSE`` — closing brackets, quotes
        - ``NON_STARTER`` — commas, periods, colons, question/exclamation marks
        """
        return atom.char_class in (CharClass.CLOSE, CharClass.NON_STARTER)

    @staticmethod
    def _is_forbidden_at_line_end(atom: Atom) -> bool:
        """Return True when *atom* must NOT appear at the end of a line.

        This covers:
        - ``OPEN`` — opening brackets, opening quotes, currency symbols
        """
        return atom.char_class == CharClass.OPEN


# ---------------------------------------------------------------------------
# Knuth-Plass optimal line breaker
# ---------------------------------------------------------------------------


class KnuthPlassLineBreaker:
    """Knuth-Plass optimal line-breaking with Kinsoku constraints.

    Uses dynamic programming to find globally optimal line breaks that
    produce even inter-character spacing across the entire paragraph.

    The algorithm minimises total "demerits" — a function of individual
    line badness (from stretching/shrinking) and breakpoint penalties
    (Kinsoku-forbidden breaks get effectively infinite penalty).

    Parameters:
        max_width: Available width per line in pixels.
        indent: First-line indent in pixels (currently unused).
        stretch_spacing: Max extra space per inter-character gap when
            stretching (pixels).  Default 10 px ≈ 0.25 em at 40 px.
        shrink_spacing: Max reduced space per gap when shrinking (pixels).
            Default 5 px ≈ 0.125 em at 40 px.
        tolerance: Maximum stretch ratio before a line is considered
            too loose.  1.0 means gaps can at most double.

    Returns from ``break_lines()``:
        ``list[tuple[int, int, float]]`` — each tuple is
        ``(start_index, end_index, adjustment_ratio)``.  The ratio tells
        the renderer::

             ratio > 0  →  stretch each gap by ratio * stretch_spacing
             ratio < 0  →  shrink each gap by ratio * shrink_spacing
             ratio ≈ 0  →  natural width (no adjustment needed)
    """

    def __init__(
        self,
        max_width: float,
        indent: float = 0.0,
        stretch_spacing: float = 10.0,
        shrink_spacing: float = 5.0,
        tolerance: float = 1.0,
    ) -> None:
        self.max_width = max_width
        self.indent = indent
        self.stretch_spacing = stretch_spacing
        self.shrink_spacing = shrink_spacing
        self.tolerance = tolerance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def break_lines(self, atoms: list[Atom]) -> list[tuple[int, int, float]]:
        """Partition *atoms* into optimally-broken lines.

        Returns:
            List of ``(start, end, adjustment_ratio)`` tuples
            (end exclusive, ratio is 0.0 for natural-width lines).
        """
        n = len(atoms)
        if n == 0:
            return []

        # Split into segments at mandatory-break boundaries.
        # Single \\n = paragraph break (no extra blank line).
        # Each additional consecutive \\n = one blank line.
        items: list[tuple[str, int, int]] = []  # ('text'|'blank', start, end)

        seg_start = 0
        idx = 0
        while idx < n:
            if atoms[idx].char_class == CharClass.MANDATORY_BREAK:
                if idx > seg_start:
                    items.append(('text', seg_start, idx))
                # Count consecutive \\n's
                nl_count = 1
                while (
                    idx + nl_count < n
                    and atoms[idx + nl_count].char_class == CharClass.MANDATORY_BREAK
                ):
                    nl_count += 1
                # Extra \\n's → blank lines (si==ei → no atoms drawn, just y advance)
                for _ in range(nl_count - 1):
                    items.append(('blank', idx, idx))
                seg_start = idx + nl_count
                idx = seg_start
            else:
                idx += 1
        if seg_start < n:
            items.append(('text', seg_start, n))

        all_lines: list[tuple[int, int, float]] = []
        for kind, si, ei in items:
            if kind == 'blank':
                all_lines.append((si, ei, 0.0))
            else:
                seg_lines = self._break_segment(atoms, si, ei)
                if seg_lines:
                    all_lines.extend(seg_lines)
                else:
                    all_lines.extend(self._fallback_segment(atoms, si, ei))

        return all_lines

    # ------------------------------------------------------------------
    # Per-segment DP
    # ------------------------------------------------------------------

    def _break_segment(
        self, atoms: list[Atom], seg_start: int, seg_end: int
    ) -> list[tuple[int, int, float]]:
        """Run Knuth-Plass DP on atoms[seg_start:seg_end] (no newlines)."""
        m = seg_end - seg_start
        if m == 0:
            return []
        if m == 1:
            return [(seg_start, seg_end, 0.0)]

        # Build valid-break array for this segment.
        # can_break[k] = can we break BEFORE atoms[seg_start + k]?
        can_break = [False] * (m + 1)
        can_break[0] = True  # segment start
        can_break[m] = True  # segment end

        for k in range(1, m):
            prev = atoms[seg_start + k - 1]
            nxt = atoms[seg_start + k]
            if nxt.char_class in (CharClass.CLOSE, CharClass.NON_STARTER):
                can_break[k] = False
            elif prev.char_class == CharClass.OPEN:
                can_break[k] = False
            else:
                can_break[k] = True

        # Prefix sums for fast width/stretch/shrink queries
        # widths[k] = sum of atoms[seg_start:seg_start+k] widths
        widths = [0.0] * (m + 1)
        for k in range(m):
            widths[k + 1] = widths[k] + atoms[seg_start + k].width

        INF = float("inf")
        best_demerits = [INF] * (m + 1)
        best_prev = [-1] * (m + 1)
        best_ratio = [0.0] * (m + 1)
        best_demerits[0] = 0.0

        stretch_sp = self.stretch_spacing
        shrink_sp = self.shrink_spacing

        for end in range(1, m + 1):
            if not can_break[end]:
                continue

            for start in range(end - 1, -1, -1):
                if not can_break[start]:
                    continue

                natural_w = widths[end] - widths[start]
                gap_count = end - start  # inter-atom gaps incl. final glue

                total_stretch = gap_count * stretch_sp
                total_shrink = gap_count * shrink_sp

                line_w = self.max_width

                # Last line of a paragraph: don't stretch, but DO shrink
                # if it would otherwise overflow.
                if end == m:
                    if natural_w > line_w:
                        if total_shrink > 0:
                            ratio = (line_w - natural_w) / total_shrink
                            if ratio < -1.0:
                                continue  # overfull even at max shrink
                        else:
                            continue  # overfull, no shrink available
                    else:
                        ratio = 0.0  # underfull last line — natural width
                    badness = 100.0 * abs(ratio) ** 3
                    penalty = 0.0
                else:
                    ratio = 0.0
                    badness = 0.0
                    penalty = 0.0

                    if natural_w < line_w:
                        if total_stretch > 0:
                            ratio = (line_w - natural_w) / total_stretch
                            if ratio > self.tolerance:
                                continue  # too loose
                    elif natural_w > line_w:
                        if total_shrink > 0:
                            ratio = (line_w - natural_w) / total_shrink
                            if ratio < -1.0:
                                continue  # overfull (can't shrink enough)

                    badness = 100.0 * abs(ratio) ** 3

                if penalty >= 0:
                    demerits = (1.0 + badness + penalty) ** 2 + best_demerits[start]
                else:
                    demerits = (1.0 + badness) ** 2 - penalty ** 2 + best_demerits[start]

                if demerits < best_demerits[end]:
                    best_demerits[end] = demerits
                    best_prev[end] = start
                    best_ratio[end] = ratio

        # Backtrack
        if best_demerits[m] >= INF:
            return []  # signal failure to caller

        lines: list[tuple[int, int, float]] = []
        end = m
        while end > 0:
            start = best_prev[end]
            if start < 0:
                return []  # corrupt path — should not happen
            ratio = best_ratio[end]
            lines.append((seg_start + start, seg_start + end, ratio))
            end = start
        lines.reverse()
        return lines

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _fallback_segment(
        self, atoms: list[Atom], seg_start: int, seg_end: int
    ) -> list[tuple[int, int, float]]:
        """Greedy fallback when DP fails on a segment."""
        breaker = KinsokuLineBreaker(self.max_width, self.indent)
        segment_atoms = atoms[seg_start:seg_end]
        raw = breaker.break_lines(segment_atoms)
        return [(seg_start + s, seg_start + e, 0.0) for s, e in raw]


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
            # Zero-width emoji modifiers (FE0F/FE0E variation selectors,
            # U+1F3FB-U+1F3FF skin tones) — keep them in the atom for
            # correct combined rendering, but measure only the base char.
            render_ch = ch  # full emoji including modifiers for rendering
            while len(ch) > 1 and (ch[-1] in '️︎'
                                   or '\U0001f3fb' <= ch[-1] <= '\U0001f3ff'):
                ch = ch[:-1]
            while offset < total and text[offset] in '️︎':
                offset += 1
            w = measure_fn(ch, font)  # width from base char only
            atom_class = CharClass.EMOJI
            # If the font can't render the full emoji (modifiers = glyph 0),
            # fall back to the base character to avoid tofu squares.
            if len(render_ch) > 1 and any(g == 0 for g in font.textToGlyphs(render_ch)):
                render_ch = ch
            atoms.append(Atom(render_ch, w, atom_class, font))
            continue

        offset += 1
        font = default_font
        w = measure_fn(ch, font)
        atom_class = classify_char(ch)
        atoms.append(Atom(ch, w, atom_class, font))

    return atoms
