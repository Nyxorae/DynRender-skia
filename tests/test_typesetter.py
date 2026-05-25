"""Tests for the typesetter module — classify_char, KinsokuLineBreaker,
KnuthPlassLineBreaker, and atomize_text."""

import pytest

from dynrender_skia.typesetter import (
    Atom,
    CharClass,
    KnuthPlassLineBreaker,
    KinsokuLineBreaker,
    classify_char,
    atomize_text,
)


# ============================================================================
# classify_char — golden tests
# ============================================================================


class TestClassifyChar:
    """Verify classify_char returns the correct CharClass for each input."""

    # ---- CJK ideographs ----
    @pytest.mark.parametrize(
        "ch",
        [
            "一",  # CJK boundary start
            "中",  # 中
            "文",  # 文
            "鿿",  # CJK boundary end
            "あ",  # Hiragana あ
            "ア",  # Katakana ア
            "가",  # Hangul 가
        ],
    )
    def test_cjk_ideograph(self, ch: str) -> None:
        assert classify_char(ch) == CharClass.IDEOGRAPH

    # ---- Space ----
    @pytest.mark.parametrize("ch", [" ", "\t"])
    def test_space(self, ch: str) -> None:
        assert classify_char(ch) == CharClass.SPACE

    def test_ideographic_space_is_ideograph(self) -> None:
        """U+3000 (ideographic space) falls in CJK range → IDEOGRAPH."""
        assert classify_char("　") == CharClass.IDEOGRAPH

    # ---- CLOSE punctuation (forbidden at line start) ----
    @pytest.mark.parametrize(
        "ch",
        [
            ")", "】", "」", "』", "》", "〉", "）",
            '"', "'", "”", "’",
        ],
    )
    def test_close_punctuation(self, ch: str) -> None:
        assert classify_char(ch) == CharClass.CLOSE

    # ---- NON_STARTER punctuation (forbidden at line start) ----
    @pytest.mark.parametrize(
        "ch",
        [
            ",", ".", "。", "、", "，", "；", "：", "?", "!", "！", "？",
            "…", "％", "%",
        ],
    )
    def test_non_starter(self, ch: str) -> None:
        assert classify_char(ch) == CharClass.NON_STARTER

    # ---- Characters in forbidden_start but classified as CLOSE (not NON_STARTER) ----
    @pytest.mark.parametrize("ch", ["℃"])
    def test_forbidden_start_close(self, ch: str) -> None:
        """℃ is in _DEFAULT_FORBIDDEN_START but not in the NON_STARTER match set."""
        assert classify_char(ch) == CharClass.CLOSE

    # ---- OPEN punctuation (forbidden at line end) ----
    @pytest.mark.parametrize(
        "ch",
        [
            "(", "【", "「", "『", "《", "〈", "（",
            "$", "￥", "£", "¥",
        ],
    )
    def test_open_punctuation(self, ch: str) -> None:
        assert classify_char(ch) == CharClass.OPEN

    # ---- OPEN quotes (also forbidden at line end) ----
    @pytest.mark.parametrize("ch", ['"', "'", "“", "‘"])
    def test_open_quotes(self, ch: str) -> None:
        # Note: ASCII quotes appear in both forbidden_start and forbidden_end
        # The _DEFAULT_FORBIDDEN_END check runs first → OPEN
        result = classify_char(ch)
        assert result in (CharClass.OPEN, CharClass.CLOSE)

    # ---- HYPHEN / dash ----
    @pytest.mark.parametrize("ch", ["-", "–", "—", "―", "‐", "‑"])
    def test_hyphen(self, ch: str) -> None:
        assert classify_char(ch) == CharClass.HYPHEN

    # ---- ALPHABETIC (fallthrough) ----
    @pytest.mark.parametrize(
        "ch",
        [
            "A", "z",  # Latin letters
            "0", "9",  # digits
            "@", "#",  # symbols
            # / + = are now in _DEFAULT_BREAKABLE_SYMBOLS → HYPHEN
            "\\",  # backslash — not in default breakable symbols
            "<", ">",  # angle brackets
            "*",  # asterisk
            "あ",  # kana small form not in forbidden list
        ],
    )
    def test_alphabetic(self, ch: str) -> None:
        # あ is in kana range → IDEOGRAPH, not ALPHABETIC
        expected = CharClass.IDEOGRAPH if ch == "あ" else CharClass.ALPHABETIC
        assert classify_char(ch) == expected

    # ---- Breakable symbols (URL separators) → HYPHEN ----
    @pytest.mark.parametrize("ch", ["/", "+", "=", "_", "&", "~"])
    def test_breakable_symbol_is_hyphen(self, ch: str) -> None:
        # ? and . are omitted — they're NON_STARTER (checked first in classify_char)
        assert classify_char(ch) == CharClass.HYPHEN

    # ---- Japanese kana small forms (in forbidden_start → CLOSE) ----
    @pytest.mark.parametrize("ch", ["ぁ", "っ", "ゃ", "ィ", "ッ", "ョ"])
    def test_small_kana(self, ch: str) -> None:
        assert classify_char(ch) == CharClass.CLOSE


# ============================================================================
# KinsokuLineBreaker tests
# ============================================================================


def _atoms(text: str, *, width: float = 10.0) -> list[Atom]:
    """Build Atom objects from a string, using classify_char for typing."""
    atoms: list[Atom] = []
    for ch in text:
        if ch == "\n":
            atoms.append(Atom(text="\n", width=0.0, char_class=CharClass.MANDATORY_BREAK))
        else:
            atoms.append(Atom(text=ch, width=width, char_class=classify_char(ch)))
    return atoms


def _text_from(atoms: list[Atom], start: int, end: int) -> str:
    return "".join(a.text for a in atoms[start:end])


class TestKinsokuLineBreaker:
    """Direct tests for the greedy Kinsoku line breaker."""

    def test_empty_atoms(self) -> None:
        breaker = KinsokuLineBreaker(max_width=100)
        assert breaker.break_lines([]) == []

    def test_single_atom(self) -> None:
        atoms = _atoms("A", width=10)
        breaker = KinsokuLineBreaker(max_width=100)
        lines = breaker.break_lines(atoms)
        assert lines == [(0, 1)]

    def test_all_fit_one_line(self) -> None:
        atoms = _atoms("Hello", width=10)
        breaker = KinsokuLineBreaker(max_width=100)
        lines = breaker.break_lines(atoms)
        assert lines == [(0, 5)]

    def test_overflow_triggers_break(self) -> None:
        atoms = _atoms("ABCDEFGHIJ", width=15)
        breaker = KinsokuLineBreaker(max_width=50)
        lines = breaker.break_lines(atoms)
        # Each atom is 15 wide, max is 50 → 3 atoms per line
        # 10 atoms → 4 lines: [0,3], [3,6], [6,9], [9,10]
        assert len(lines) == 4
        assert lines[0] == (0, 3)
        assert lines[1] == (3, 6)
        assert lines[-1] == (9, 10)
        # Verify no line exceeds max_width
        for s, e in lines:
            line_w = sum(a.width for a in atoms[s:e])
            assert line_w <= 50

    def test_mandatory_break(self) -> None:
        atoms = _atoms("AB\nCD", width=10)
        breaker = KinsokuLineBreaker(max_width=100)
        lines = breaker.break_lines(atoms)
        # \n is consumed (i advances past it) and included in the line range.
        # The renderer skips MANDATORY_BREAK atoms when drawing.
        assert lines == [(0, 3), (3, 5)]
        assert _text_from(atoms, *lines[0]) == "AB\n"
        assert _text_from(atoms, *lines[1]) == "CD"

    def test_consecutive_newlines(self) -> None:
        atoms = _atoms("A\n\nB", width=10)
        breaker = KinsokuLineBreaker(max_width=100)
        lines = breaker.break_lines(atoms)
        # Atoms: A(0), \n(1), \n(2), B(3)
        # Fill 1: A fits (10px), \n at idx 1 → i=2, break → line (0,2) = "A\n"
        # Fill 2: \n at idx 2 → i=3, break → line (2,3) = "\n"
        # Fill 3: B at idx 3 → fill to end → line (3,4) = "B"
        assert len(lines) == 3
        assert _text_from(atoms, *lines[0]) == "A\n"
        assert _text_from(atoms, *lines[1]) == "\n"
        assert _text_from(atoms, *lines[2]) == "B"

    # ---- Kinsoku pull-back (追い出し) ----

    def test_pull_back_closing_bracket(self) -> None:
        """Closing bracket at line start must be pulled back."""
        # "AAAAB" → A is 10px, B is 10px, max 45px: 4 chars fit
        # 5th char 'B' would start line 2. Not a forbidden-char case.
        # Instead test: "AAAA。」BBB" with 10px each, max 50px.
        # 5 chars fit. "。」" are CLOSE/NON_STARTER. If break after AAAA,
        # next line starts with "。」" → pull-back needed.
        atoms = [
            Atom("A", 10, CharClass.IDEOGRAPH),
            Atom("A", 10, CharClass.IDEOGRAPH),
            Atom("A", 10, CharClass.IDEOGRAPH),
            Atom("A", 10, CharClass.IDEOGRAPH),
            Atom("A", 10, CharClass.IDEOGRAPH),
            Atom("。", 10, CharClass.NON_STARTER),
            Atom("」", 10, CharClass.CLOSE),
            Atom("B", 10, CharClass.IDEOGRAPH),
            Atom("B", 10, CharClass.IDEOGRAPH),
            Atom("B", 10, CharClass.IDEOGRAPH),
        ]
        breaker = KinsokuLineBreaker(max_width=50)
        lines = breaker.break_lines(atoms)
        # Verify no line starts with CLOSE or NON_STARTER
        for s, e in lines:
            assert atoms[s].char_class not in (CharClass.CLOSE, CharClass.NON_STARTER)
        # Verify all atoms accounted for
        total = sum(e - s for s, e in lines)
        assert total == len(atoms)

    def test_pull_back_comma(self) -> None:
        """Comma (NON_STARTER) must not appear at line start."""
        atoms = [
            Atom("A", 15, CharClass.IDEOGRAPH),
            Atom("B", 15, CharClass.IDEOGRAPH),
            Atom("C", 15, CharClass.IDEOGRAPH),
            Atom("，", 15, CharClass.NON_STARTER),  # will overflow → index 3
            Atom("D", 15, CharClass.IDEOGRAPH),
        ]
        breaker = KinsokuLineBreaker(max_width=45)
        lines = breaker.break_lines(atoms)
        # Verify no line starts with NON_STARTER
        for s, e in lines:
            if e > s:
                assert atoms[s].char_class != CharClass.NON_STARTER

    # ---- Kinsoku push-forward (追い込み) ----

    def test_push_forward_open_bracket(self) -> None:
        """Open bracket at line end must be pushed forward."""
        atoms = [
            Atom("A", 15, CharClass.IDEOGRAPH),
            Atom("B", 15, CharClass.IDEOGRAPH),
            Atom("（", 15, CharClass.OPEN),  # at index 2, forbidden at line end
            Atom("C", 15, CharClass.IDEOGRAPH),
            Atom("D", 15, CharClass.IDEOGRAPH),
        ]
        breaker = KinsokuLineBreaker(max_width=45)
        lines = breaker.break_lines(atoms)
        # Verify no line ends with OPEN
        for s, e in lines:
            if e > s:
                assert atoms[e - 1].char_class != CharClass.OPEN

    # ---- Safety valve ----

    def test_single_wide_atom_forces_one_per_line(self) -> None:
        """When one atom alone overflows, the safety valve (j <= i → j = i+1) kicks in."""
        atoms = [
            Atom("W", 200, CharClass.IDEOGRAPH),  # wider than max
            Atom("X", 10, CharClass.IDEOGRAPH),
        ]
        breaker = KinsokuLineBreaker(max_width=100)
        lines = breaker.break_lines(atoms)
        assert len(lines) == 2
        assert lines[0] == (0, 1)  # forced single atom
        assert lines[1] == (1, 2)

    # ---- Combined kinsoku ----

    def test_kinsoku_chain_open_close(self) -> None:
        """「text」 — open and close should stay with their content."""
        atoms = [
            Atom("A", 15, CharClass.IDEOGRAPH),
            Atom("A", 15, CharClass.IDEOGRAPH),
            Atom("「", 15, CharClass.OPEN),
            Atom("X", 15, CharClass.IDEOGRAPH),
            Atom("」", 15, CharClass.CLOSE),
            Atom("B", 15, CharClass.IDEOGRAPH),
            Atom("B", 15, CharClass.IDEOGRAPH),
        ]
        breaker = KinsokuLineBreaker(max_width=60)
        lines = breaker.break_lines(atoms)
        # 「 should not end a line, 」 should not start a line
        for s, e in lines:
            if e > s:
                assert atoms[e - 1].char_class != CharClass.OPEN
                assert atoms[s].char_class not in (CharClass.CLOSE, CharClass.NON_STARTER)

    # ---- Invariant: all atoms covered in order ----

    def test_invariant_all_atoms_covered(self) -> None:
        atoms = _atoms("Hello, 世界！This is a test.「禁則処理」の確認。", width=12)
        breaker = KinsokuLineBreaker(max_width=80)
        lines = breaker.break_lines(atoms)
        covered = 0
        prev_end = 0
        for s, e in lines:
            assert s == prev_end, f"gap or overlap at line {s}: prev_end={prev_end}"
            covered += e - s
            prev_end = e
        assert covered == len(atoms)


# ============================================================================
# KnuthPlassLineBreaker tests
# ============================================================================


class TestKnuthPlassLineBreaker:
    """Tests for the Knuth-Plass optimal line-breaking algorithm."""

    def test_empty_atoms(self) -> None:
        breaker = KnuthPlassLineBreaker(max_width=100)
        assert breaker.break_lines([]) == []

    def test_single_atom(self) -> None:
        atoms = _atoms("A", width=10)
        breaker = KnuthPlassLineBreaker(max_width=100)
        lines = breaker.break_lines(atoms)
        assert len(lines) == 1
        assert lines[0][0] == 0
        assert lines[0][1] == 1

    def test_single_short_line(self) -> None:
        atoms = _atoms("Hello", width=10)
        breaker = KnuthPlassLineBreaker(max_width=100)
        lines = breaker.break_lines(atoms)
        assert len(lines) == 1
        s, e, ratio = lines[0]
        # Underfull last line → ratio should be 0 (natural width)
        assert ratio == 0.0
        assert (s, e) == (0, 5)

    def test_multiple_lines_no_kinsoku(self) -> None:
        """Plain ASCII text — DP should produce balanced breaks."""
        atoms = _atoms("ABCDEFGHIJKLMNOPQRSTUVWXYZ", width=10)
        breaker = KnuthPlassLineBreaker(max_width=50, stretch_spacing=5, shrink_spacing=3)
        lines = breaker.break_lines(atoms)
        # 26 chars * 10px = 260px → ~6 lines at 50px each (5 chars per line)
        assert len(lines) >= 5
        # Invariant: all atoms covered
        total = sum(e - s for s, e, _ in lines)
        assert total == len(atoms)
        # Invariant: no line width exceeds max_width (at max shrink)
        for s, e, ratio in lines:
            line_w = sum(a.width for a in atoms[s:e])
            n_gaps = e - s
            if n_gaps > 1 and ratio < 0:
                line_w += ratio * 3 * n_gaps  # shrink_spacing=3
            assert line_w <= 55, f"line [{s}:{e}] width {line_w:.0f} > 50"

    def test_mandatory_break_splits_segment(self) -> None:
        atoms = _atoms("AB\nCD", width=10)
        breaker = KnuthPlassLineBreaker(max_width=100)
        lines = breaker.break_lines(atoms)
        # two segments: "AB" and "CD"
        assert len(lines) == 2
        assert _text_from(atoms, lines[0][0], lines[0][1]) == "AB"
        assert _text_from(atoms, lines[1][0], lines[1][1]) == "CD"

    def test_consecutive_newlines_produce_blank_lines(self) -> None:
        atoms = _atoms("A\n\nB", width=10)
        breaker = KnuthPlassLineBreaker(max_width=100)
        lines = breaker.break_lines(atoms)
        # "A", blank, "B"
        # blank lines have start==end
        blank_lines = [(s, e, r) for s, e, r in lines if s == e]
        assert len(blank_lines) == 1
        # All atoms covered (A + \n + \n + B = 4 atoms, \n consumed as breaks)
        total = sum(e - s for s, e, _ in lines)
        assert total == 2  # only "A" and "B" drawn

    def test_kinsoku_constrained_break(self) -> None:
        """DP should respect kinsoku: no CLOSE at line start, no OPEN at line end."""
        atoms = [
            Atom("A", 15, CharClass.IDEOGRAPH),
            Atom("A", 15, CharClass.IDEOGRAPH),
            Atom("（", 15, CharClass.OPEN),
            Atom("X", 15, CharClass.IDEOGRAPH),
            Atom("）", 15, CharClass.CLOSE),
            Atom("B", 15, CharClass.IDEOGRAPH),
            Atom("B", 15, CharClass.IDEOGRAPH),
            Atom("B", 15, CharClass.IDEOGRAPH),
        ]
        breaker = KnuthPlassLineBreaker(max_width=60, stretch_spacing=5, shrink_spacing=3)
        lines = breaker.break_lines(atoms)
        for s, e, _ in lines:
            if e > s:
                assert atoms[e - 1].char_class != CharClass.OPEN, f"OPEN at end of line [{s}:{e}]"
                assert atoms[s].char_class not in (
                    CharClass.CLOSE, CharClass.NON_STARTER,
                ), f"CLOSE/NON_STARTER at start of line [{s}:{e}]"

    def test_fallback_on_impossible_break(self) -> None:
        """When even max shrink can't fit, expect fallback (no crash)."""
        # One very wide atom: can't break within it
        atoms = [
            Atom("X", 200, CharClass.IDEOGRAPH),
            Atom("Y", 10, CharClass.IDEOGRAPH),
        ]
        breaker = KnuthPlassLineBreaker(max_width=100)
        lines = breaker.break_lines(atoms)
        # Should not crash; may fall back to greedy
        assert len(lines) > 0
        total = sum(e - s for s, e, _ in lines)
        assert total == len(atoms)

    def test_ratio_is_zero_for_last_line(self) -> None:
        """Last line of a paragraph should have ratio=0 (no stretching).

        Uses CJK ideographs to avoid mid-word penalties that skew the DP.
        """
        atoms = _atoms("中文测试内容展示", width=12)
        breaker = KnuthPlassLineBreaker(max_width=50, stretch_spacing=10, shrink_spacing=5)
        lines = breaker.break_lines(atoms)
        assert len(lines) >= 2
        _, _, last_ratio = lines[-1]
        assert last_ratio == 0.0

    def test_invariant_sequential_coverage(self) -> None:
        """Lines must cover atoms in order without gaps or overlaps."""
        atoms = _atoms(
            "Hello, 世界！This is a test of the Knuth-Plass line breaker. "
            "It should produce balanced lines with even spacing. "
            "禁則処理も正しく動作すること。",
            width=14,
        )
        breaker = KnuthPlassLineBreaker(max_width=120, stretch_spacing=8, shrink_spacing=4)
        lines = breaker.break_lines(atoms)
        prev_end = 0
        for s, e, _ in lines:
            assert s == prev_end, f"gap/overlap at [{s}:{e}] prev_end={prev_end}"
            prev_end = e
        assert prev_end == len(atoms)


# ============================================================================
# atomize_text tests
# ============================================================================


class TestAtomizeText:
    """Tests for atomize_text — converting raw text + emoji info into Atoms."""

    class _MockFont:
        """A minimal mock that supports textToGlyphs (returns non-zero glyphs)."""

        def textToGlyphs(self, text: str) -> list[int]:
            return [1] * len(text)

    @staticmethod
    def _measure(ch: str, font: object) -> float:
        """Dummy measure: 10px per char (emoji = 20px for distinction)."""
        return 20.0 if len(ch.encode("utf-8")) > 3 else 10.0

    @staticmethod
    def _mock_font() -> _MockFont:
        """Return a mock font with textToGlyphs support."""
        return TestAtomizeText._MockFont()

    def test_plain_ascii(self) -> None:
        atoms = atomize_text("ABC", self._measure, {}, self._mock_font(), self._mock_font())
        assert len(atoms) == 3
        assert [a.text for a in atoms] == ["A", "B", "C"]
        assert all(a.char_class == CharClass.ALPHABETIC for a in atoms)

    def test_cjk_text(self) -> None:
        atoms = atomize_text("中文测试", self._measure, {}, self._mock_font(), self._mock_font())
        assert len(atoms) == 4
        assert all(a.char_class == CharClass.IDEOGRAPH for a in atoms)

    def test_newline_creates_mandatory_break(self) -> None:
        atoms = atomize_text("A\nB", self._measure, {}, self._mock_font(), self._mock_font())
        assert len(atoms) == 3
        assert atoms[1].char_class == CharClass.MANDATORY_BREAK
        assert atoms[1].width == 0.0

    def test_emoji_in_map(self) -> None:
        emoji_map = {5: [6, "😊"]}  # offset 5 → end 6
        atoms = atomize_text("Hello😊", self._measure, emoji_map, self._mock_font(), self._mock_font())
        # "Hello" = 5 chars + 1 emoji = 6 atoms
        assert len(atoms) == 6
        assert atoms[5].char_class == CharClass.EMOJI

    def test_emoji_not_in_map_treated_as_text(self) -> None:
        """When offset not in emoji_map, emoji chars are classified normally."""
        atoms = atomize_text("😀", self._measure, {}, self._mock_font(), self._mock_font())
        assert len(atoms) == 1
        # Emoji char not in map → goes through classify_char → likely ALPHABETIC
        # (unless it's in forbidden sets)

    def test_mixed_cjk_and_ascii(self) -> None:
        atoms = atomize_text("Hello世界", self._measure, {}, self._mock_font(), self._mock_font())
        assert len(atoms) == 7  # 5 ASCII + 2 CJK
        assert atoms[0].char_class == CharClass.ALPHABETIC  # H
        assert atoms[5].char_class == CharClass.IDEOGRAPH  # 世

    def test_punctuation_classified_correctly(self) -> None:
        atoms = atomize_text("。、（）", self._measure, {}, self._mock_font(), self._mock_font())
        assert atoms[0].char_class == CharClass.NON_STARTER  # 。
        assert atoms[1].char_class == CharClass.NON_STARTER  # 、
        assert atoms[2].char_class == CharClass.OPEN  # （
        assert atoms[3].char_class == CharClass.CLOSE  # ）

    def test_zwj_emoji_kept_as_single_atom(self) -> None:
        """ZWJ family emoji (7 codepoints) must be treated as one atom."""
        # Simulate what emoji.emoji_list returns for 👨‍👩‍👧‍👦
        family = "👨‍👩‍👧‍👦"  # 7 codepoints
        emoji_map = {0: (7, family)}
        text = family + "!"
        atoms = atomize_text(text, self._measure, emoji_map, self._mock_font(), self._mock_font())
        # 1 emoji atom + 1 punctuation atom = 2 total
        assert len(atoms) == 2
        assert atoms[0].char_class == CharClass.EMOJI
        assert atoms[0].text == family  # full sequence preserved
        assert len(atoms[0].text) == 7  # all 7 codepoints

    def test_skin_tone_emoji_kept_intact(self) -> None:
        """Skin tone modifier must be part of the emoji atom."""
        emoji_with_tone = "👨🏻"  # man + light skin tone, 2 codepoints
        emoji_map = {0: (2, emoji_with_tone)}
        atoms = atomize_text(emoji_with_tone, self._measure, emoji_map, self._mock_font(), self._mock_font())
        assert len(atoms) == 1
        assert atoms[0].char_class == CharClass.EMOJI
        assert atoms[0].text == emoji_with_tone  # both codepoints preserved
