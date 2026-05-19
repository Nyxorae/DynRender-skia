"""Additional-card renderers — registered via decorator-based strategy pattern."""

import re
from os import path
from typing import Optional

import numpy as np
import skia
from dynamicadaptor.AddonCard import Additional

from loguru import logger

from ..config import PolyStyle
from ..graphics import TextDrawer, fetch_images, paste, draw_shadow, round_corners, make_badge
from .registry import register_additional


class BaseAdditionalRenderer:
    """Base for additional-card renderers."""

    def __init__(self, src_path: str, style: PolyStyle, additional: Additional):
        self.src_path = src_path
        self.style = style
        self.additional = additional
        self.canvas: Optional[skia.Canvas] = None
        self.text_font = skia.Font(
            skia.Typeface.MakeFromName(style.font.font_family, style.font.font_style),
            style.font.font_size.text,
        )
        self.emoji_font = skia.Font(
            skia.Typeface.MakeFromName(style.font.emoji_font_family, style.font.font_style),
            style.font.font_size.text,
        )
        self._drawer = TextDrawer(style)

    async def _draw_shadow(self, pos, corner, bg_color):
        await draw_shadow(self.canvas, pos, corner, bg_color)

    async def _round_corners(self, img, corner):
        return await round_corners(img, corner)

    async def _draw_text(self, text, font_size, pos, font_color, font_style=None):
        await self._drawer.draw_text(self.canvas, text, font_size, pos, font_color, font_style)

    async def _make_badge(self, text, font_size, pos, img_size, text_pos):
        font = self.text_font
        if font.textToGlyphs(text=text[0])[0] == 0:
            if typeface := skia.FontMgr().matchFamilyStyleCharacter(
                self.style.font.font_family, self.style.font.font_style, ["zh", "en"], ord(text[0]),
            ):
                font = skia.Font(typeface, self.style.font.font_size.text)
        await make_badge(self.canvas, text, font, font_size,
                         self.style.color.font_color.name_big_vip, pos, img_size, text_pos)

    def _bg(self, repost):
        return self.style.color.background.repost if repost else self.style.color.background.normal


@register_additional("ADDITIONAL_TYPE_RESERVE")
class AdditionalReserve(BaseAdditionalRenderer):
    async def run(self, repost=False):
        bg = self._bg(repost)
        surface = skia.Surface(1080, 225)
        self.canvas = surface.getCanvas()
        self.canvas.clear(skia.Color(*bg))
        try:
            await self._draw_shadow((35, 20, 1010, 185), 15, bg)
            await self._make_desc()
            await self._make_badge("预约", self.style.font.font_size.text, (850, 75), (170, 75), (45, 50))
            return self.canvas.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)
        except Exception as e:
            logger.exception(e)
            return None

    async def _make_desc(self):
        if self.additional.reserve.desc3 is not None:
            await self._draw_text(self.additional.reserve.title, self.style.font.font_size.time,
                                  (75, 70, 740, 70, 0), self.style.color.font_color.text)
            desc_1 = f"{self.additional.reserve.desc1.text}  {self.additional.reserve.desc2.text}"
            await self._draw_text(desc_1, self.style.font.font_size.title,
                                  (75, 120, 810, 120, 0), self.style.color.font_color.sub_title)
            await self._draw_text(self.additional.reserve.desc3.text, self.style.font.font_size.title,
                                  (105, 170, 810, 170, 0), self.style.color.font_color.rich_text)
            lottery_img = skia.Image.open(path.join(self.src_path, "lottery.png")).resize(40, 40)
            await paste(self.canvas, lottery_img, (65, 138))
        else:
            await self._draw_text(self.additional.reserve.title, self.style.font.font_size.time,
                                  (75, 100, 740, 100, 0), self.style.color.font_color.text)
            desc_1 = f"{self.additional.reserve.desc1.text}  {self.additional.reserve.desc2.text}"
            await self._draw_text(desc_1, self.style.font.font_size.title,
                                  (75, 160, 810, 160, 0), self.style.color.font_color.sub_title)


@register_additional("ADDITIONAL_TYPE_UPOWER_LOTTERY")
class AdditionalUpowerLottery(BaseAdditionalRenderer):
    async def run(self, repost=False):
        bg = self._bg(repost)
        surface = skia.Surface(1080, 225)
        self.canvas = surface.getCanvas()
        self.canvas.clear(skia.Color(*bg))
        try:
            await self._draw_shadow((35, 20, 1010, 185), 15, bg)
            await self._make_desc()
            await self._make_badge("去看看", self.style.font.font_size.time, (860, 75), (155, 75), (25, 50))
            return self.canvas.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)
        except Exception as e:
            logger.exception(e)
            return None

    async def _make_desc(self):
        await self._draw_text(self.additional.upower_lottery.title, self.style.font.font_size.time,
                              (75, 100, 740, 100, 0), self.style.color.font_color.text)
        await self._draw_text(self.additional.upower_lottery.desc.text, self.style.font.font_size.title,
                              (105, 160, 810, 160, 0), self.style.color.font_color.rich_text)
        lottery_img = skia.Image.open(path.join(self.src_path, "lottery.png")).resize(40, 40)
        await paste(self.canvas, lottery_img, (65, 128))


@register_additional("ADDITIONAL_TYPE_GOODS")
class AdditionalGoods(BaseAdditionalRenderer):
    async def run(self, repost=False):
        bg = self._bg(repost)
        surface = skia.Surface(1080, 310)
        self.canvas = surface.getCanvas()
        self.canvas.clear(skia.Color(*bg))
        try:
            await self._draw_shadow((35, 50, 1010, 240), 15, bg)
            await self._make_cover()
            await self._make_title_desc()
            await self._draw_text(self.additional.goods.head_text, self.style.font.font_size.sub_title,
                                  (45, 30, 1010, 80, 0), self.style.color.font_color.sub_title)
            return self.canvas.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)
        except Exception as e:
            logger.exception(e)
            return None

    async def _make_cover(self):
        url_list = []
        for i in self.additional.goods.items:
            url = re.sub(r"@(\d+)h_(\d+)w\S+", "", i.cover)
            url_list.append(f"{url}@160w_160h_1c.webp")
        covers = await fetch_images(url_list, (190, 190))
        if len(covers) > 1:
            for i, j in enumerate(covers):
                x = 45 + i * 200
                if x > 1000:
                    break
                await paste(self.canvas, await round_corners(j, 10), (x, 75))
        else:
            await paste(self.canvas, await round_corners(covers[0], 10), (60, 75))
            await self._make_badge("去看看", self.style.font.font_size.time, (860, 125), (155, 75), (25, 50))

    async def _make_title_desc(self):
        if len(self.additional.goods.items) > 1:
            return
        await self._draw_text(self.additional.goods.items[0].name, self.style.font.font_size.title,
                              (275, 140, 800, 140, 0), self.style.color.font_color.text)
        price = f"{self.additional.goods.items[0].price}起"
        await self._draw_text(price, self.style.font.font_size.title,
                              (295, 210, 800, 210, 0), self.style.color.font_color.rich_text)


@register_additional("ADDITIONAL_TYPE_UGC")
class AdditionalUgc(BaseAdditionalRenderer):
    async def run(self, repost=False):
        bg = self._bg(repost)
        surface = skia.Surface(1080, 280)
        self.canvas = surface.getCanvas()
        self.canvas.clear(skia.Color(*bg))
        try:
            await self._draw_shadow((35, 20, 1010, 240), 15, bg)
            rec = skia.Rect.MakeXYWH(35, 20, 1010, 240)
            self.canvas.clipRRect(skia.RRect(rec, 20, 20), skia.ClipOp.kIntersect)
            await self._make_cover()
            await self._make_title_desc()
            await self._make_sub_tag()
            return self.canvas.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)
        except Exception as e:
            logger.exception(e)
            return None

    async def _make_cover(self):
        cover = await fetch_images(f"{self.additional.ugc.cover}@340w_195h_1c.webp")
        await paste(self.canvas, await round_corners(cover, 10), (60, 45))

    async def _make_title_desc(self):
        await self._draw_text(self.additional.ugc.title, self.style.font.font_size.title,
                              (430, 90, 990, 140, int(self.style.font.font_size.time * 1.3)),
                              self.style.color.font_color.text)
        await self._draw_text(self.additional.ugc.desc_second, self.style.font.font_size.title,
                              (430, 220, 950, 220, 0), self.style.color.font_color.sub_title)

    async def _make_sub_tag(self):
        self.text_font.setSize(self.style.font.font_size.sub_title)
        size = self.text_font.measureText(self.additional.ugc.duration)
        surface = skia.Surface(int(size + 20), int(self.text_font.getSize() + 20))
        canvas = surface.getCanvas()
        canvas.clear(skia.Color(0, 0, 0, 150))
        blob = skia.TextBlob(self.additional.ugc.duration, self.text_font)
        paint = skia.Paint(AntiAlias=True, Color=skia.Color4f.kWhite)
        canvas.drawTextBlob(blob, 10, int(self.text_font.getSize() + 5), paint)
        sub_tag_img = skia.Image.fromarray(
            canvas.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType),
            colorType=skia.ColorType.kRGBA_8888_ColorType,
        )
        await paste(self.canvas, await round_corners(sub_tag_img, 10),
                    (400 - sub_tag_img.width() - 15, 190))


@register_additional("ADDITIONAL_TYPE_VOTE")
class AdditionalVote(BaseAdditionalRenderer):
    async def run(self, repost=False):
        bg = self._bg(repost)
        surface = skia.Surface(1080, 280)
        self.canvas = surface.getCanvas()
        self.canvas.clear(skia.Color(*bg))
        try:
            await self._draw_shadow((35, 20, 1010, 240), 15, bg)
            rec = skia.Rect.MakeXYWH(35, 20, 1010, 240)
            self.canvas.clipRRect(skia.RRect(rec, 20, 20), skia.ClipOp.kIntersect)
            await self._make_cover()
            await self._make_title_desc()
            await self._make_badge("投票", self.style.font.font_size.time, (860, 95), (155, 75), (42, 50))
            return self.canvas.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)
        except Exception as e:
            logger.exception(e)
            return None

    async def _make_cover(self):
        cover = skia.Image.open(fp=path.join(self.src_path, "vote_icon.png")).resize(195, 195)
        await paste(self.canvas, await round_corners(cover, 10), (60, 45))

    async def _make_title_desc(self):
        await self._draw_text(self.additional.vote.desc, self.style.font.font_size.text,
                              (280, 110, 780, 110, 0), self.style.color.font_color.text)
        join_num = f"{self.additional.vote.join_num}人参与" if self.additional.vote.join_num else "0人参与"
        await self._draw_text(join_num, self.style.font.font_size.time,
                              (280, 190, 780, 190, 0), self.style.color.font_color.sub_title)


@register_additional("ADDITIONAL_TYPE_COMMON")
class AdditionalCommon(BaseAdditionalRenderer):
    async def run(self, repost=False):
        bg = self._bg(repost)
        surface = skia.Surface(1080, 340)
        self.canvas = surface.getCanvas()
        self.canvas.clear(skia.Color(*bg))
        try:
            await self._draw_shadow((35, 80, 1010, 245), 15, bg)
            await self._draw_text(self.additional.common.head_text, self.style.font.font_size.title,
                                  (50, 50, 1010, 90, 0), self.style.color.font_color.sub_title)
            rec = skia.Rect.MakeXYWH(35, 80, 1010, 240)
            self.canvas.clipRRect(skia.RRect(rec, 20, 20), skia.ClipOp.kIntersect)
            await self._make_cover()
            await self._select_badge()
            await self._make_title()
            await self._make_desc()
            return self.canvas.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)
        except Exception as e:
            logger.exception(e)
            return None

    async def _make_cover(self):
        if self.additional.common.sub_type in {"decoration", "game"}:
            cover_url = f"{self.additional.common.cover}@190w_190h_1c.webp"
            cover = await fetch_images(cover_url, (190, 190))
        else:
            cover_url = f"{self.additional.common.cover}@145w_195h_1c.webp"
            cover = await fetch_images(cover_url, (145, 195))
        await paste(self.canvas, await round_corners(cover, 15), (60, 110))

    async def _make_title(self):
        y = 150 if self.additional.common.desc2 else 180
        x = 280 if self.additional.common.sub_type in {"decoration", "game"} else 250
        await self._draw_text(self.additional.common.title, self.style.font.font_size.text,
                              (x, y, 780, y, 0), self.style.color.font_color.text)

    async def _make_desc(self):
        x = 280 if self.additional.common.sub_type in {"decoration", "game"} else 250
        if self.additional.common.desc2:
            await self._draw_text(self.additional.common.desc1, self.style.font.font_size.title,
                                  (x, 220, 780, 160, 0), self.style.color.font_color.sub_title)
            await self._draw_text(self.additional.common.desc2, self.style.font.font_size.title,
                                  (x, 285, 780, 225, 0), self.style.color.font_color.sub_title)
        else:
            await self._draw_text(self.additional.common.desc1, self.style.font.font_size.title,
                                  (x, 250, 780, 190, 0), self.style.color.font_color.sub_title)

    async def _select_badge(self):
        badge_map = {"pugv": "去试看", "ogv": "去看看", "manga": "去追漫", "decoration": "去看看", "game": "进入"}
        badge_text = badge_map.get(self.additional.common.sub_type, "去看看")
        size = self.text_font.measureText(badge_text)
        x = int((155 - size) / 2)
        await self._make_badge(badge_text, self.style.font.font_size.time, (860, 165), (155, 75), (x, 50))
