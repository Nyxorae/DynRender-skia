"""Shape and decoration helpers — corners, shadows, cropping, badges."""

import numpy as np
import skia

from .composite import paste


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
