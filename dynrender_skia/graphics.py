"""Common graphics primitives: canvas operations, text drawing, image fetching/merging."""

import asyncio
from os import path
from typing import Optional, Union

import httpx
import numpy as np
import skia
from loguru import logger
from numpy import ndarray

from .config import PolyStyle

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


# async def merge_pictures(img_list: list[ndarray]) -> ndarray:
#     """Vertically stack image arrays into one.

#     Pre-allocates the output array when possible (avoids the O(n²)
#     copying of repeated ``vstack`` calls).
#     """
#     # Fast path: single image
#     if len(img_list) == 1 and img_list[0] is not None:
#         return img_list[0]

#     # Filter None entries and validate widths
#     valid = [img for img in img_list if img is not None]
#     if not valid:
#         return np.zeros([0, 1080, 4], np.uint8)
#     for img in valid:
#         if img.shape[1] != 1080:
#             raise ValueError("The width of the image must be 1080")

#     # Pre-allocate and copy in one pass
#     total_height = sum(img.shape[0] for img in valid)
#     result = np.zeros([total_height, 1080, 4], np.uint8)
#     offset = 0
#     for img in valid:
#         h = img.shape[0]
#         result[offset:offset + h] = img
#         offset += h
#     return result

async def merge_pictures(img_list: list[ndarray]) -> ndarray:
    """Vertically stack image arrays into one."""
    # 保留原始行为：单元素列表且非 None 直接返回（不做复制，也不检查宽度）
    if len(img_list) == 1 and img_list[0] is not None:
        return img_list[0]

    # 一次遍历：过滤 None、检查宽度、收集有效图像
    valid = []
    for img in img_list:
        if img is None:
            continue
        if img.shape[1] != 1080:
            raise ValueError("The width of the image must be 1080")
        valid.append(img)

    if not valid:
        return np.zeros([0, 1080, 4], dtype=np.uint8)

    # 如果只剩下一个有效图像（且原列表不只一个元素），为保持原行为仍做拷贝
    # 用 concatenate 一次性拼接，内部自动预分配 + 拷贝
    return np.concatenate(valid, axis=0)

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
# Text drawing (unified engine) — lives in its own module for separation
# ---------------------------------------------------------------------------

from .text_drawer import TextDrawer  # noqa: F401 — re-export for backward compat
