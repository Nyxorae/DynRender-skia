"""Common graphics primitives: canvas operations, text drawing, image fetching/merging."""

import asyncio
from os import path
from typing import Optional, Union

import emoji
import httpx
import numpy as np
import skia
from loguru import logger
from numpy import ndarray

from .config import PolyStyle
from .exceptions import ParseError

# ---------------------------------------------------------------------------
# Image fetching — shared connection pool for performance
# ---------------------------------------------------------------------------

_IMG_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Referer": "https://t.bilibili.com/",
    "Origin": "https://t.bilibili.com",
}

# Shared client avoids creating a new TCP connection pool per request.
# Created lazily on first use; closed on process exit.
_client: Optional[httpx.AsyncClient] = None
_client_lock = asyncio.Lock()


async def _get_client() -> httpx.AsyncClient:
    """Return the module-level shared HTTP client (lazy init, thread-safe)."""
    global _client
    if _client is not None and not _client.is_closed:
        return _client
    async with _client_lock:
        if _client is None or _client.is_closed:
            _client = httpx.AsyncClient(
                transport=httpx.AsyncHTTPTransport(retries=5),
                headers=_IMG_HEADERS,
                timeout=httpx.Timeout(30.0),
            )
        return _client


async def fetch_images(
    url: Union[str, list[str]], size: Optional[tuple[int, int]] = None, retries: int = 5
) -> Union[skia.Image, tuple[skia.Image, ...]]:
    """Fetch image(s) from URL(s), optionally resizing.

    Uses a shared HTTP client for connection pooling — orders of magnitude
    faster than creating a new client per call.
    """
    client = await _get_client()
    if isinstance(url, list):
        return await asyncio.gather(
            *[_request_img(client, u, size) for u in url]
        )
    return await _request_img(client, url, size)


async def _request_img(
    client: httpx.AsyncClient, url: str, size: Optional[tuple[int, int]],
) -> Optional[skia.Image]:
    try:
        response = await client.get(url)
        img: skia.Image = skia.Image.MakeFromEncoded(response.content)  # type: ignore
        if img is None:
            logger.error("Image decode error or request returned none in content")
        return img.resize(*size) if size is not None else img
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        logger.exception(f"Request or HTTP error occurred: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return None


# Keep the old name for backward compatibility (tests use it)
request_img = _request_img


# ---------------------------------------------------------------------------
# Image merging
# ---------------------------------------------------------------------------


async def merge_pictures(img_list: list[ndarray]) -> ndarray:
    """Vertically stack image arrays into one.

    Pre-allocates the output array when possible (avoids the O(n²)
    copying of repeated ``vstack`` calls).
    """
    # Fast path: single image
    if len(img_list) == 1 and img_list[0] is not None:
        return img_list[0]

    # Filter None entries and validate widths
    valid = [img for img in img_list if img is not None]
    if not valid:
        return np.zeros([0, 1080, 4], np.uint8)
    for img in valid:
        if img.shape[1] != 1080:
            raise ValueError("The width of the image must be 1080")

    # Pre-allocate and copy in one pass
    total_height = sum(img.shape[0] for img in valid)
    result = np.zeros([total_height, 1080, 4], np.uint8)
    offset = 0
    for img in valid:
        h = img.shape[0]
        result[offset:offset + h] = img
        offset += h
    return result


# ---------------------------------------------------------------------------
# Canvas paste
# ---------------------------------------------------------------------------


async def paste(canvas: skia.Canvas, target: skia.Image, position: tuple, clear_background: bool = False) -> None:
    x, y = position
    img_height = target.dimensions().fHeight
    img_width = target.dimensions().fWidth
    rec = skia.Rect.MakeXYWH(x, y, img_width, img_height)  # type: ignore
    try:
        if clear_background:
            canvas.save()
            canvas.clipRect(rec, skia.ClipOp.kIntersect)
            canvas.clear(skia.Color(*(255, 255, 255, 0)))
        canvas.drawImageRect(target, skia.Rect(0, 0, img_width, img_height), rec)
        if clear_background:
            canvas.restore()
    except AttributeError as e:
        logger.exception(f"Failed to paste image: {e!s}")


# ---------------------------------------------------------------------------
# Canvas builder — flattens the repetitive surface→clear→shadow→clip flow
# ---------------------------------------------------------------------------


class CanvasBuilder:
    """Builder pattern for Skia canvas construction.

    Usage::

        builder = CanvasBuilder(1080, 695).with_background(bg).with_shadow(
            (35, 25, 1010, 655), 20, bg
        ).with_clip((35, 25, 1010, 665), 20)
        canvas = builder.build()
        # ... draw on canvas ...
        arr = builder.to_array()
    """

    def __init__(self, width: int, height: int) -> None:
        self._surface = skia.Surface(width, height)
        self._canvas = self._surface.getCanvas()
        self._bg_color: tuple = (255, 255, 255, 255)

    def with_background(self, color: tuple) -> "CanvasBuilder":
        """Clear the canvas with *color* (RGBA tuple)."""
        self._canvas.clear(skia.Color(*color))
        self._bg_color = color
        return self

    def with_shadow(self, rect: tuple, corner: int, bg_color: tuple) -> "CanvasBuilder":
        """Draw a rounded rectangle with a drop-shadow.

        Args:
            rect: ``(x, y, width, height)`` of the shadow area.
            corner: Corner radius in pixels.  0 = sharp rectangle.
            bg_color: Background color (the shadow is cast FROM this).
        """
        self._sync_shadow(rect, corner, bg_color)
        return self

    def with_clip(self, rect: tuple, corner: int) -> "CanvasBuilder":
        """Clip subsequent drawing to a rounded rectangle.

        Args:
            rect: ``(x, y, width, height)``.
            corner: Corner radius.
        """
        rec = skia.Rect.MakeXYWH(*rect)
        self._canvas.clipRRect(skia.RRect(rec, corner, corner), skia.ClipOp.kIntersect)
        return self

    def build(self) -> skia.Canvas:
        """Return the constructed canvas."""
        return self._canvas

    def to_array(self):
        """Export the canvas contents as an RGBA numpy array."""
        return self._canvas.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)

    # -- internal ----------------------------------------------------------

    async def _async_shadow(self, rect: tuple, corner: int, bg_color: tuple) -> None:
        await draw_shadow(self._canvas, rect, corner, bg_color)

    def _sync_shadow(self, rect: tuple, corner: int, bg_color: tuple) -> None:
        """Synchronous shadow — use when ``asyncio`` is not needed."""
        x, y, w, h = rect
        rec = skia.Rect.MakeXYWH(x, y, w, h)
        paint = skia.Paint(
            Color=skia.Color(*bg_color),
            AntiAlias=True,
            ImageFilter=skia.ImageFilters.DropShadow(0, 0, 10, 10, skia.Color(120, 120, 120)),
        )
        if corner != 0:
            self._canvas.drawRoundRect(rec, corner, corner, paint)
        else:
            self._canvas.drawRect(rec, paint)


# ---------------------------------------------------------------------------
# Shape / decoration helpers
# ---------------------------------------------------------------------------


async def round_corners(img: skia.Image, corner: int) -> skia.Image:
    surface = skia.Surface(img.width(), img.height())
    mask = surface.getCanvas()
    paint = skia.Paint(
        Style=skia.Paint.kFill_Style,
        Color=skia.Color(255, 255, 255, 255),
        AntiAlias=True,
    )
    rect = skia.Rect.MakeXYWH(0, 0, img.width(), img.height())
    mask.drawRoundRect(rect, corner, corner, paint)
    image_array = np.bitwise_and(
        img.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType),
        mask.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType),
    )
    return skia.Image.fromarray(image_array, colorType=skia.ColorType.kRGBA_8888_ColorType)


async def draw_shadow(
    canvas: skia.Canvas, pos: tuple, corner: int, bg_color: tuple
) -> None:
    x, y, width, height = pos
    rec = skia.Rect.MakeXYWH(x, y, width, height)
    paint = skia.Paint(
        Color=skia.Color(*bg_color),
        AntiAlias=True,
        ImageFilter=skia.ImageFilters.DropShadow(0, 0, 10, 10, skia.Color(120, 120, 120)),
    )
    if corner != 0:
        canvas.drawRoundRect(rec, corner, corner, paint)
    else:
        canvas.drawRect(rec, paint)


async def circle_crop(img: skia.Image, size: int, ring_color: tuple = (251, 114, 153, 255)) -> skia.Image:
    w, h = img.dimensions().width(), img.dimensions().height()
    surface = skia.Surface(w, h)
    mask = surface.getCanvas()
    fill = skia.Paint(Style=skia.Paint.kFill_Style, Color=skia.Color(255, 255, 255, 255), AntiAlias=True)
    stroke = skia.Paint(Style=skia.Paint.kStroke_Style, StrokeWidth=5, Color=skia.Color(*ring_color), AntiAlias=True)
    radius = int(w / 2)
    mask.drawCircle(radius, radius, radius, fill)
    image_array = np.bitwise_and(
        img.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType),
        mask.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType),
    )
    canvas = skia.Canvas(image_array, colorType=skia.ColorType.kRGBA_8888_ColorType)
    canvas.drawCircle(radius, radius, radius - 2, stroke)
    return skia.Image.fromarray(
        canvas.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType),
        colorType=skia.ColorType.kRGBA_8888_ColorType,
    ).resize(size, size)  # type: ignore


async def make_badge(
    canvas: skia.Canvas,
    text: str,
    font: skia.Font,
    font_size: int,
    bg_color: tuple,
    pos: tuple,
    img_size: tuple,
    text_pos: tuple,
) -> None:
    font.setSize(font_size)
    surface = skia.Surface(*img_size)
    badge_canvas = surface.getCanvas()
    badge_canvas.clear(skia.Color(*bg_color))
    blob = skia.TextBlob(text=text, font=font)  # type: ignore
    paint = skia.Paint(AntiAlias=True, Color=skia.Color4f.kWhite)
    badge_canvas.drawTextBlob(blob, text_pos[0], text_pos[1], paint)
    tag_img = skia.Image.fromarray(
        array=badge_canvas.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType),
        colorType=skia.ColorType.kRGBA_8888_ColorType,
    )  # type: ignore
    tag_img = await round_corners(tag_img, 10)
    await paste(canvas, tag_img, pos)


async def make_sub_tag(
    canvas: skia.Canvas, text: str, font: skia.Font, font_size: int, pos: tuple
) -> None:
    font.setSize(font_size)
    size = font.measureText(text)
    surface = skia.Surface(int(size + 20), int(font.getSize() + 20))
    tag_canvas = surface.getCanvas()
    tag_canvas.clear(skia.Color(0, 0, 0, 150))
    blob = skia.TextBlob(text, font)
    paint = skia.Paint(AntiAlias=True, Color=skia.Color4f.kWhite)
    tag_canvas.drawTextBlob(blob, 10, int(font.getSize() + 5), paint)
    img = skia.Image.fromarray(
        tag_canvas.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType),
        colorType=skia.ColorType.kRGBA_8888_ColorType,
    )
    await paste(canvas, await round_corners(img, 10), pos)


# ---------------------------------------------------------------------------
# Text drawing (unified engine)
# ---------------------------------------------------------------------------


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

        Unlike :meth:`_resolver.resolve`, this does NOT fall back to
        ``self.text_font`` — it returns ``None`` when the system has no
        match, preserving the original API contract expected by tests.
        """
        if typeface := skia.FontMgr().matchFamilyStyleCharacter(
            self.style.font.font_family,
            self.style.font.font_style,
            ["zh", "en"],
            ord(char),
        ):
            return skia.Font(typeface, font_size)
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
        """True when *font* can render *char* (glyph ID ≠ 0)."""
        glyphs = font.textToGlyphs(char)
        return len(glyphs) > 0 and glyphs[0] != 0

    @staticmethod
    def _needs_new_line(x: int, max_w: int) -> bool:
        """True when *x* exceeds the text area width, triggering a line break."""
        return x > max_w

    def _advance_to_next_line(self, current_y: int, line_spacing: int, max_height: int, initial_x: int,
                   canvas: skia.Canvas, font: skia.Font, paint: skia.Paint, current_x: int) -> tuple[int, int]:
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

        tf_id = self.text_font.getTypeface().uniqueID()
        ef_id = self.emoji_font.getTypeface().uniqueID()

        from .typesetter import atomize_text, KinsokuLineBreaker, CharClass

        def measure(ch: str, font: skia.Font) -> float:
            fid = font.getTypeface().uniqueID()
            cache_key = (ch, font_size, fid)
            if cache_key in self._width_cache:
                return self._width_cache[cache_key]
            if self._font_contains_character(font, ch):
                w = font.measureText(ch)
            else:
                font_key = (ch, font_size)
                if font_key not in self._font_cache:
                    resolved = self.match_font(ch, font_size)
                    self._font_cache[font_key] = resolved or font
                w = self._font_cache[font_key].measureText(ch)
            self._width_cache[cache_key] = w
            return w

        atoms = atomize_text(text, measure, emoji_info, self.text_font, self.emoji_font)
        if not atoms:
            return

        breaker = KinsokuLineBreaker(max_width=x_bound - start_x, indent=0)
        lines = breaker.break_lines(atoms)

        current_y = start_y
        for line_idx, (si, ei) in enumerate(lines):
            if line_idx > 0 and current_y >= y_bound:
                self.draw_ellipsis(canvas, last_x, current_y - line_spacing,
                                   self.text_font, paint)
                break
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
                canvas.drawTextBlob(skia.TextBlob(atom.text, font), current_x, current_y, paint)
                current_x += atom.width
            last_x = current_x
            current_y += line_spacing
