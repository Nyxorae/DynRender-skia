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
    # Middle dot, en-dash, URL separator — can break before/after but prefers not to separate
    HYPHEN = auto()


# ---------------------------------------------------------------------------
# Kinsoku configuration — makes forbidden sets and penalties externally configurable
# ---------------------------------------------------------------------------

# Default forbidden-start characters (CLOSE + NON_STARTER per UAX #14 + GB/T 15834)
_DEFAULT_FORBIDDEN_START: set[str] = {
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

_DEFAULT_FORBIDDEN_END: set[str] = {
    # Open brackets — CJK and western
    "(", "【", "〔", "［", "「", "『", "《", "〈", "（",
    # Open quotes
    "\"", "'", "“", "‘", "「", "『",
    # Currency symbols that precede numbers
    "$", "￥", "£", "¥",
}

_DEFAULT_HYPHEN_CHARS: set[str] = {"-", "‐", "‑", "‒", "–", "—", "―"}

# URL / path separators — breakable so long URLs don't overflow
_DEFAULT_BREAKABLE_SYMBOLS: set[str] = {"/", "_", "&", "=", "~", "+"}


@dataclass
class KinsokuConfig:
    """Configurable kinsoku (禁則処理) rules and break-penalty parameters.

    All fields have defaults matching the CJK standard rules.  Pass an
    instance to ``KinsokuLineBreaker`` or ``KnuthPlassLineBreaker`` to
    customise behaviour for a specific platform or language variant.

    Attributes:
        forbidden_start: Characters that MUST NOT appear at line-start.
        forbidden_end: Characters that MUST NOT appear at line-end.
        breakable_symbols: URL/path separators where breaking is preferred
            over mid-word.  Classified as ``HYPHEN``.
        max_push_distance: Max characters to push-forward (追い込み).
        max_pull_distance: Max characters to pull-back (追い出し).
        penalty_mid_word: Penalty for breaking between two ALPHABETIC chars
            (discouraged but allowed for overflow).
        penalty_hyphen: Penalty for breaking at a HYPHEN / breakable symbol.
        penalty_sentence_end: Penalty for breaking after sentence-ending
            NON_STARTER punctuation (negative = preferred).
    """

    forbidden_start: set[str] = field(default_factory=lambda: _DEFAULT_FORBIDDEN_START.copy())
    forbidden_end: set[str] = field(default_factory=lambda: _DEFAULT_FORBIDDEN_END.copy())
    breakable_symbols: set[str] = field(default_factory=lambda: _DEFAULT_BREAKABLE_SYMBOLS.copy())
    max_push_distance: int = 3
    max_pull_distance: int = 3
    penalty_mid_word: float = 50.0
    penalty_hyphen: float = 10.0
    penalty_sentence_end: float = 0.0

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


# Non-starter subset (punctuation that is NON_STARTER, not CLOSE)
_NON_STARTER_SUBSET: set[str] = {
    ",", ".", "。", "、", "，", "．", "；", "：", "?", "!", "！", "？",
    "…", "％", "%",
}


def classify_char(ch: str, config: Optional[KinsokuConfig] = None) -> CharClass:
    """Classify a character (or compound emoji) for line-breaking purposes.

    Multi-character strings (emoji + variation selector) are classified
    by their first code-point only.

    Args:
        ch: The character (or multi-codepoint grapheme) to classify.
        config: Optional KinsokuConfig for custom forbidden sets and
            breakable symbols.  When None, uses the built-in defaults.
    """
    f_start = config.forbidden_start if config else _DEFAULT_FORBIDDEN_START
    f_end = config.forbidden_end if config else _DEFAULT_FORBIDDEN_END
    breakable = config.breakable_symbols if config else _DEFAULT_BREAKABLE_SYMBOLS

    if ch in f_start:
        return CharClass.NON_STARTER if ch in _NON_STARTER_SUBSET else CharClass.CLOSE
    if ch in f_end:
        return CharClass.OPEN
    cp = ord(ch)
    if _is_cjk(cp):
        return CharClass.IDEOGRAPH
    if ch.isspace():
        return CharClass.SPACE
    if ch in _DEFAULT_HYPHEN_CHARS or ch in breakable:
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

    def __init__(
        self,
        max_width: float,
        indent: float = 0.0,
        config: Optional[KinsokuConfig] = None,
    ) -> None:
        self.max_width = max_width
        self.indent = indent
        self.config = config or KinsokuConfig()
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

        When overflow occurs inside an ASCII word, walks backward to the
        last space or hyphen for a more readable break.  Stops at
        MANDATORY_BREAK (``\\n``) — the break character is consumed but
        not included in the line.
        """
        width = 0.0
        i = start
        while i < n:
            atom = atoms[i]
            if atom.char_class == CharClass.MANDATORY_BREAK:
                i += 1       # consume the newline
                break
            if width + atom.width > max_w and width > 0:
                # Overflow on an ALPHABETIC char — try to break at the last
                # space or hyphen within this word for readability.
                if atom.char_class == CharClass.ALPHABETIC:
                    lookback = i - 1
                    while lookback > start:
                        prev_class = atoms[lookback].char_class
                        if prev_class in (CharClass.SPACE, CharClass.HYPHEN):
                            # Break AFTER the space/hyphen → it stays on line end
                            return lookback + 1
                        if prev_class != CharClass.ALPHABETIC:
                            break  # different script — stop looking
                        lookback -= 1
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

        Pull-back stops after ``config.max_pull_distance`` steps to
        prevent pulling the entire line contents.
        """
        max_dist = self.config.max_pull_distance
        pulls = 0
        while break_at < n and break_at > line_start and pulls < max_dist:
            first_of_next = atoms[break_at]
            if not self._is_forbidden_at_line_start(first_of_next):
                break
            break_at -= 1
            pulls += 1
            # Continue pulling if the preceding char is forbidden at line-end.
            # Handles chains like  "《text》"  → 《 must not end a line.
            while break_at > line_start and pulls < max_dist and self._is_forbidden_at_line_end(
                atoms[break_at - 1]
            ):
                break_at -= 1
                pulls += 1
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

        Stops after ``config.max_push_distance`` steps.
        """
        max_dist = self.config.max_push_distance
        pushes = 0
        while break_at > line_start and break_at <= n and pushes < max_dist:
            last_of_line = atoms[break_at - 1]
            if not self._is_forbidden_at_line_end(last_of_line):
                break
            break_at -= 1
            pushes += 1
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
        config: Optional[KinsokuConfig] = None,
    ) -> None:
        self.max_width = max_width
        self.indent = indent
        self.stretch_spacing = stretch_spacing
        self.shrink_spacing = shrink_spacing
        self.tolerance = tolerance
        self.config = config or KinsokuConfig()

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
                    from loguru import logger

                    logger.warning(
                        "Knuth-Plass line breaking failed for segment "
                        "[{}:{}] ({} atoms), falling back to KinsokuLineBreaker",
                        si, ei, ei - si,
                    )
                    all_lines.extend(self._fallback_segment(atoms, si, ei))

        return all_lines

    # ------------------------------------------------------------------
    # Per-segment DP
    # ------------------------------------------------------------------

    def _break_segment(
        self, atoms: list[Atom], seg_start: int, seg_end: int
    ) -> list[tuple[int, int, float]]:
        """Run Knuth-Plass DP on atoms[seg_start:seg_end] (no newlines).

        Penalties from ``self.config`` shape breakpoint preferences:
        - Mid-word breaks (ALPHABETIC→ALPHABETIC): penalty_mid_word (default +50)
        - Hyphen / breakable symbol: penalty_hyphen (default +10)
        - Sentence-end breaks (after NON_STARTER): penalty_sentence_end (default -50)
        - Forbidden breaks (CLOSE/NON_STARTER at line-start, OPEN at line-end):
          hard-blocked via can_break.
        """
        m = seg_end - seg_start
        if m == 0:
            return []
        if m == 1:
            return [(seg_start, seg_end, 0.0)]

        # Build valid-break and penalty arrays for this segment.
        # can_break[k] = can we break BEFORE atoms[seg_start + k]?
        can_break = [False] * (m + 1)
        break_penalty = [0.0] * (m + 1)
        can_break[0] = True  # segment start
        can_break[m] = True  # segment end

        cfg = self.config

        for k in range(1, m):
            prev = atoms[seg_start + k - 1]
            nxt = atoms[seg_start + k]

            # Hard constraints — kinsoku-forbidden breaks
            if nxt.char_class in (CharClass.CLOSE, CharClass.NON_STARTER):
                can_break[k] = False
                continue
            if prev.char_class == CharClass.OPEN:
                can_break[k] = False
                continue

            can_break[k] = True

            # Soft penalty — discourage mid-word breaks, prefer sentence-end
            if prev.char_class == CharClass.ALPHABETIC and nxt.char_class == CharClass.ALPHABETIC:
                break_penalty[k] = cfg.penalty_mid_word
            elif prev.char_class in (CharClass.HYPHEN,) or nxt.char_class in (CharClass.HYPHEN,):
                break_penalty[k] = cfg.penalty_hyphen
            elif prev.char_class == CharClass.NON_STARTER:
                break_penalty[k] = cfg.penalty_sentence_end

        # Prefix sums for fast width/stretch/shrink queries
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
                gap_count = end - start

                total_stretch = gap_count * stretch_sp
                total_shrink = gap_count * shrink_sp

                line_w = self.max_width

                if end == m:
                    if natural_w > line_w:
                        if total_shrink > 0:
                            ratio = (line_w - natural_w) / total_shrink
                            if ratio < -1.0:
                                continue
                        else:
                            continue
                    else:
                        ratio = 0.0
                    badness = 100.0 * abs(ratio) ** 3
                    penalty = 0.0  # last line: no break penalty (matching TeX)
                else:
                    ratio = 0.0
                    badness = 0.0

                    if natural_w < line_w:
                        if total_stretch > 0:
                            ratio = (line_w - natural_w) / total_stretch
                            if ratio > self.tolerance:
                                continue
                    elif natural_w > line_w:
                        if total_shrink > 0:
                            ratio = (line_w - natural_w) / total_shrink
                            if ratio < -1.0:
                                continue

                    badness = 100.0 * abs(ratio) ** 3
                    penalty = break_penalty[start]

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
            # *emoji_char* is the full sequence from emoji.emoji_list(),
            # which correctly handles ZWJ (U+200D), variation selectors
            # (FE0F/FE0E), and skin-tone modifiers (U+1F3FB–U+1F3FF).
            render_ch = emoji_char
            # Strip trailing zero-width modifiers from *emoji_char* for
            # width measurement — they affect rendering but not metrics.
            base = emoji_char
            while len(base) > 1 and (
                base[-1] in '️︎'
                or '\U0001f3fb' <= base[-1] <= '\U0001f3ff'
            ):
                base = base[:-1]
            # Skip any trailing variation selectors beyond the emoji map end
            while offset < total and text[offset] in '️︎':
                offset += 1
            w = measure_fn(base, font)
            atom_class = CharClass.EMOJI
            # If the font can't render the full emoji sequence (ZWJ /
            # modifiers = glyph 0), fall back to the base chars to avoid
            # tofu squares.
            if len(render_ch) > 1 and any(g == 0 for g in font.textToGlyphs(render_ch)):
                render_ch = base
            atoms.append(Atom(render_ch, w, atom_class, font))
            continue

        offset += 1
        font = default_font
        w = measure_fn(ch, font)
        atom_class = classify_char(ch)
        atoms.append(Atom(ch, w, atom_class, font))

    return atoms
