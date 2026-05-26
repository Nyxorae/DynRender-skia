"""Footer renderer."""

from os import path
from time import localtime, strftime, time
from typing import Optional

import numpy as np
import skia
from loguru import logger

from ..config import PolyStyle
from ..graphics import TextDrawer


class Footer:
    """Render the footer timestamp."""

    def __init__(self, static_path: str, style: PolyStyle) -> None:
        self.src_path = path.join(static_path, "Src")
        self.style = style
        self._drawer = TextDrawer(style)

    async def run(self) -> Optional[np.ndarray]:
        surface = skia.Surface(1080, 110)
        canvas = surface.getCanvas()
        canvas.clear(skia.Color(*self.style.color.background.normal))
        try:
            now = strftime("%Y-%m-%d %H:%M:%S", localtime(time()))
            render_time = f"图片生成于：{now}"
            await self._drawer.draw_text(
                canvas,
                render_time,
                self.style.font.font_size.title,
                (35, 70, 1010, 70, 0),
                self.style.color.font_color.sub_title,
            )
            return canvas.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)
        except Exception as e:
            logger.exception(e)
            return None
