"""Image compositing — merge and paste operations."""

from typing import Optional

import numpy as np
import skia
from loguru import logger
from numpy import ndarray


async def merge_pictures(img_list: list[Optional[ndarray]]) -> ndarray:
    """Vertically stack image arrays into one."""
    if len(img_list) == 1 and img_list[0] is not None:
        return img_list[0]

    valid = []
    for img in img_list:
        if img is None:
            continue
        if img.shape[1] != 1080:
            raise ValueError("The width of the image must be 1080")
        valid.append(img)

    if not valid:
        return np.zeros([0, 1080, 4], dtype=np.uint8)

    return np.concatenate(valid, axis=0)


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
