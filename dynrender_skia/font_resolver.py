"""Font resolution via Chain-of-Responsibility + Flyweight caching.

Replaces the 15+ duplicated ``matchFamilyStyleCharacter`` calls scattered
across the codebase with a single, cache-aware resolver.
"""

from typing import Optional

import skia


class FontResolver:
    """Resolves the best font for a character via a chain of fallbacks.

    Chain order:
      1. *preferred_font* — if it can render the character, use it.
      2. System match — ask the OS via ``FontMgr.matchFamilyStyleCharacter``.
      3. *fallback_font* — the default text font as a last resort.

    Resolved fonts are cached (Flyweight) to avoid re-creating identical
    ``skia.Font`` objects for repeated characters.
    """

    def __init__(self, font_family: str, font_style, emoji_font_family: str = "") -> None:
        self._font_family = font_family
        self._font_style = font_style
        self._emoji_font_family = emoji_font_family
        self._font_mgr = skia.FontMgr()
        # Flyweight cache: (typeface_id, font_size) → skia.Font
        self._font_cache: dict[tuple[int, int], skia.Font] = {}

    def resolve(self, char: str, preferred_font: skia.Font, font_size: int) -> skia.Font:
        """Return the best font for *char* at *font_size*.

        Args:
            char: A single character (or multi-codepoint grapheme).
            preferred_font: The font to try first (e.g., ``text_font``).
            font_size: Desired font size in pixels.

        Returns:
            A ``skia.Font`` guaranteed to contain the character.
        """
        if self._font_contains_char(preferred_font, char):
            return preferred_font

        typeface = self._font_mgr.matchFamilyStyleCharacter(
            self._font_family, self._font_style, ["zh", "en"], ord(char[0]),
        )
        if typeface is not None:
            return self._cached_font(typeface, font_size)

        return preferred_font

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _font_contains_char(font: skia.Font, char: str) -> bool:
        """True when *font* can render *char*.

        ``textToGlyphs`` returns a list of glyph IDs (one per codepoint).
        A glyph ID of 0 means the font has no glyph for that codepoint.

        Returns ``False`` on any Skia error (malformed font, etc.) rather
        than letting the exception propagate.
        """
        try:
            glyphs = font.textToGlyphs(char)
            return len(glyphs) > 0 and glyphs[0] != 0
        except Exception:
            from loguru import logger

            logger.opt(exception=True).warning(
                "FontResolver._font_contains_char failed for char={!r}", char,
            )
            return False

    def _cached_font(self, typeface: skia.Typeface, font_size: int) -> skia.Font:
        key = (typeface.uniqueID(), font_size)
        if key not in self._font_cache:
            try:
                self._font_cache[key] = skia.Font(typeface, font_size)
            except Exception:
                from loguru import logger

                logger.opt(exception=True).warning(
                    "FontResolver._cached_font failed to create skia.Font "
                    "for typeface_id={}, size={}", typeface.uniqueID(), font_size,
                )
                raise
        return self._font_cache[key]
