"""Unified text layout and drawing engine.

Extracted from ``graphics.py`` to keep the graphics module focused on
pure Skia primitives (images, shapes, compositing).
"""

from typing import Optional

import emoji
import skia
from loguru import logger

from .config import PolyStyle
from .exceptions import ParseError


class TextDrawer:
    """Unified text layout and drawing engine.

    Uses :class:`FontResolver` (Chain-of-Responsibility) to find the
    best font for each character.

    **Performance optimisations:**

    - Character-width cache (LRU).  CJK texts repeat the same characters
      thousands of times; avoiding ``measureText`` for known widths is
      the single biggest text-rendering win.
    - Resolved fonts are cached keyed by ``(char, font_size)``.

    Public API — callers use :meth:`draw_text` for kinsoku-aware
    rendering, or the static helpers for one-off paint/badge creation.
    """

    def __init__(self, style: PolyStyle):
        self.style = style
        self.text_font = skia.Font(
            skia.Typeface.MakeFromName(style.font.font_family, style.font.font_style),
            style.font.font_size.text,
        )
        self.emoji_font = skia.Font(
            skia.Typeface.MakeFromName(style.font.emoji_font_family, style.font.font_style),
            style.font.font_size.text,
        )
        from .font_resolver import FontResolver

        self._resolver = FontResolver(
            style.font.font_family, style.font.font_style, style.font.emoji_font_family,
        )
        # Width cache: (char, font_size, font_id) -> width (float)
        self._width_cache: dict[tuple[str, int, int], float] = {}
        # Resolved font cache: (char, font_size) -> skia.Font
        self._font_cache: dict[tuple[str, int], skia.Font] = {}

    # ------------------------------------------------------------------
    # Static helpers (used by tests)
    # ------------------------------------------------------------------

    @staticmethod
    def initialize_paint(font_color: tuple) -> skia.Paint:
        """Create an anti-aliased ``skia.Paint`` filled with *font_color*."""
        return skia.Paint(AntiAlias=True, Color=skia.Color(*font_color))

    @staticmethod
    def draw_ellipsis(
        canvas: skia.Canvas, x: int, y: int, font: skia.Font, paint: skia.Paint,
    ) -> None:
        """Draw a "…" truncation marker at *(x, y)*."""
        canvas.drawTextBlob(skia.TextBlob("...", font), x, y, paint)

    # ------------------------------------------------------------------
    # Font helpers
    # ------------------------------------------------------------------

    def match_font(self, char: str, font_size: int) -> Optional[skia.Font]:
        """Try to find a system font that can render *char*.

        First tries the configured font family, then falls back to
        any available system font (broad search).  Returns ``None``
        only when no font on the system supports the character (or if
        the font manager raises an exception).
        """
        try:
            for family in (self.style.font.font_family, ""):
                if typeface := skia.FontMgr().matchFamilyStyleCharacter(
                    family, self.style.font.font_style, ["zh", "en"], ord(char[0]),
                ):
                    return skia.Font(typeface, font_size)
        except Exception:
            logger.opt(exception=True).warning(
                "TextDrawer.match_font failed for char={!r} (U+{:04X}), size={}",
                char, ord(char[0]), font_size,
            )
        return None

    def set_font_sizes(self, size: int) -> None:
        """Set both the text and emoji font sizes to *size*."""
        self.text_font.setSize(size)
        self.emoji_font.setSize(size)

    @staticmethod
    async def get_emoji_text(text: str) -> dict[int, list]:
        result = emoji.emoji_list(text)
        return {i["match_start"]: [i["match_end"], i["emoji"]] for i in result}

    async def extract_emoji_info(self, text: str) -> tuple[str, dict[int, list]]:
        text = text.replace("\t", "")
        emoji_info = await self.get_emoji_text(text)
        return text, emoji_info

    @staticmethod
    def _font_contains_character(font: skia.Font, char: str) -> bool:
        """True when *font* can render *char* (glyph ID ≠ 0).

        Returns ``False`` on any Skia error rather than crashing.
        """
        try:
            glyphs = font.textToGlyphs(char)
            return len(glyphs) > 0 and glyphs[0] != 0
        except Exception:
            return False

    @staticmethod
    def _needs_new_line(x: int, max_w: int) -> bool:
        """True when *x* exceeds the text area width, triggering a line break."""
        return x > max_w

    def _advance_to_next_line(
        self,
        current_y: int,
        line_spacing: int,
        max_height: int,
        initial_x: int,
        canvas: skia.Canvas,
        font: skia.Font,
        paint: skia.Paint,
        current_x: int,
    ) -> tuple[int, int]:
        if current_y + line_spacing >= max_height:
            self.draw_ellipsis(canvas, current_x, current_y, font, paint)
            return max_height, initial_x
        return current_y + line_spacing, initial_x

    def _handle_emoji(self, offset: int, emoji_info: dict[int, list]) -> tuple[int, str, skia.Font]:
        try:
            character = emoji_info[offset][1]
            end_pos = emoji_info[offset][0]
            return end_pos, character, self.emoji_font
        except KeyError as e:
            raise ParseError(f"Error parsing emoji information {e}") from e

    async def _get_emoji_info(self, text: str) -> dict[int, list]:
        return await self.get_emoji_text(text)

    async def draw_text(
        self,
        canvas: skia.Canvas,
        text: str,
        font_size: int,
        pos: tuple,
        font_color: tuple,
        font_style=None,
    ):
        paint = self.initialize_paint(font_color)
        if font_style is not None:
            self.text_font = skia.Font(
                skia.Typeface.MakeFromName(self.style.font.font_family, font_style),
                self.style.font.font_size.text,
            )
        self.set_font_sizes(font_size)

        text = text.replace("\t", "")
        emoji_info = await self.get_emoji_text(text)
        start_x, start_y, x_bound, y_bound, line_spacing = pos

        from .typesetter import atomize_text, KnuthPlassLineBreaker, CharClass

        def measure(ch: str, font: skia.Font) -> float:
            fid = font.getTypeface().uniqueID()
            cache_key = (ch, font_size, fid)
            if cache_key in self._width_cache:
                return self._width_cache[cache_key]
            try:
                if self._font_contains_character(font, ch):
                    w = font.measureText(ch)
                else:
                    font_key = (ch, font_size)
                    if font_key not in self._font_cache:
                        resolved = self.match_font(ch, font_size)
                        self._font_cache[font_key] = resolved or font
                    w = self._font_cache[font_key].measureText(ch)
            except Exception:
                logger.opt(exception=True).warning(
                    "TextDrawer.measure failed for char={!r} font_size={}", ch, font_size,
                )
                w = font_size  # fallback width ≈ 1 em
            self._width_cache[cache_key] = w
            return w

        atoms = atomize_text(text, measure, emoji_info, self.text_font, self.emoji_font)
        if not atoms:
            return

        max_w = x_bound - start_x
        breaker = KnuthPlassLineBreaker(max_width=max_w, indent=0)
        lines = breaker.break_lines(atoms)

        current_y = start_y
        ellipsis_w = self.text_font.measureText("...")

        for line_idx, (si, ei, _ratio) in enumerate(lines):
            # If the next line would overflow y_bound, this line is the
            # last visible one — leave room for the ellipsis.
            last_visible = current_y + line_spacing >= y_bound

            current_x = start_x
            for k in range(si, ei):
                atom = atoms[k]
                if atom.char_class == CharClass.MANDATORY_BREAK:
                    continue
                font = atom.font or self.text_font
                if not self._font_contains_character(font, atom.text):
                    font_key = (atom.text, font_size)
                    if font_key not in self._font_cache:
                        self._font_cache[font_key] = self.match_font(atom.text, font_size) or font
                    font = self._font_cache[font_key]
                draw_text = atom.text
                blob = None
                if atom.char_class == CharClass.EMOJI and len(atom.text) > 1:
                    if any(g == 0 for g in font.textToGlyphs(atom.text)):
                        draw_text = atom.text[0]
                    else:
                        # HarfBuzz shaping to combine modifiers with base emoji
                        blob = skia.TextBlob.MakeFromShapedText(atom.text, font)
                        canvas.drawTextBlob(blob, current_x, current_y, paint)
                        current_x += atom.width
                        # Last-visible check after advancing x
                        if last_visible and current_x + ellipsis_w > x_bound:
                            self.draw_ellipsis(canvas, current_x, current_y,
                                               self.text_font, paint)
                            return
                        continue
                if blob is None:
                    blob = skia.TextBlob(draw_text, font)
                next_x = current_x + atom.width
                if last_visible and next_x + ellipsis_w > x_bound:
                    self.draw_ellipsis(canvas, current_x, current_y,
                                       self.text_font, paint)
                    return
                canvas.drawTextBlob(blob, current_x, current_y, paint)
                current_x = next_x

            if last_visible:
                # Only draw ellipsis if more lines exist below that won't fit
                if line_idx < len(lines) - 1:
                    ex = min(current_x, x_bound - ellipsis_w)
                    self.draw_ellipsis(canvas, ex, current_y,
                                       self.text_font, paint)
                return
            current_y += line_spacing
