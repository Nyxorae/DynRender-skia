"""Template-method base class for renderers."""

from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np
import skia
from loguru import logger

from ..config import PolyStyle
from ..exceptions import DynRenderException, SkiaBaseError
from ..graphics import TextDrawer, draw_shadow, paste, round_corners


async def safe_run(fn: Callable, *args, **kwargs) -> Optional[np.ndarray]:
    """Execute a render function with unified error handling.

    Distinguishes between known render errors (SkiaBaseError) and
    unexpected exceptions, logging at different levels accordingly.
    """
    try:
        return await fn(*args, **kwargs)
    except SkiaBaseError as e:
        logger.error(f"Render aborted ({type(e).__name__}): {e}")
        return None
    except DynRenderException as e:
        logger.error(f"Render aborted ({type(e).__name__}): {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected render failure: {e}")
        return None


class BaseRenderer(ABC):
    """Base renderer with template-method pattern for the common render flow.

    Subclasses override _render() and optionally _init_canvas().
    """

    def __init__(self, src_path: str, style: PolyStyle):
        self.src_path = src_path
        self.style = style
        self.canvas: Optional[skia.Canvas] = None
        self.text_font = skia.Font(
            skia.Typeface.MakeFromName(self.style.font.font_family, self.style.font.font_style),
            self.style.font.font_size.text,
        )
        self.emoji_font = skia.Font(
            skia.Typeface.MakeFromName(self.style.font.emoji_font_family, self.style.font.font_style),
            self.style.font.font_size.text,
        )
        self._drawer = TextDrawer(style)
        self._surface: Optional[skia.Surface] = None

    def _init_canvas(self, width: int, height: int, bg_color: tuple) -> None:
        self._surface = skia.Surface(width, height)
        self.canvas = self._surface.getCanvas()
        self.canvas.clear(skia.Color(*bg_color))

    async def _draw_shadow(self, pos: tuple, corner: int, bg_color: tuple) -> None:
        await draw_shadow(self.canvas, pos, corner, bg_color)

    async def _round_corners(self, img: skia.Image, corner: int) -> skia.Image:
        return await round_corners(img, corner)

    async def _paste(self, target: skia.Image, position: tuple, clear_background: bool = False) -> None:
        await paste(self.canvas, target, position, clear_background)

    async def _draw_text(self, text: str, font_size: int, pos: tuple, font_color: tuple, font_style=None) -> None:
        await self._drawer.draw_text(self.canvas, text, font_size, pos, font_color, font_style)

    async def _get_emoji_info(self, text: str) -> dict:
        return await self._drawer._get_emoji_info(text)

    def _to_array(self) -> Optional[np.ndarray]:
        if self.canvas is None:
            return None
        return self.canvas.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)

    @abstractmethod
    async def run(self, data, repost: bool = False) -> Optional[np.ndarray]:
        ...
