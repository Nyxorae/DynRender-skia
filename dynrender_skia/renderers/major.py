"""Major content renderers — registered via decorator-based strategy pattern."""

from math import ceil
from os import path
from typing import Optional

import numpy as np
import skia
from dynamicadaptor.Content import RichTextDetail, Text
from dynamicadaptor.Majors import Major, RichTextNodes
from loguru import logger

from ..config import PolyStyle
from ..font_resolver import FontResolver
from ..graphics import TextDrawer, draw_shadow, fetch_images, merge_pictures, paste, round_corners
from .registry import register_major
from .text import BiliText


class BaseMajorRenderer:
    """Template-method base for all major-type renderers.

    Provides:
    - Canvas helpers: ``_draw_shadow``, ``_round_corners``, ``_draw_text``
    - Badge/tag rendering: ``_make_tag``, ``_make_sub_tag``
    - Font resolution via :class:`FontResolver` (Chain-of-Responsibility)
    - Background color selection via ``_bg(repost)``

    Subclasses override ``run(repost)`` and return a numpy array or None.
    """

    def __init__(self, src_path: str, style: PolyStyle, major: Major = None):
        self.src_path = src_path
        self.style = style
        self.major = major
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
        self._resolver = FontResolver(
            style.font.font_family, style.font.font_style, style.font.emoji_font_family,
        )

    async def _draw_shadow(self, pos, corner, bg_color):
        await draw_shadow(self.canvas, pos, corner, bg_color)

    async def _round_corners(self, img, corner):
        return await round_corners(img, corner)

    async def _draw_text(self, text, font_size, pos, font_color, font_style=None):
        await self._drawer.draw_text(self.canvas, text, font_size, pos, font_color, font_style)

    async def _make_tag(self, tag: str, font_size: int):
        """Draw a pink badge (e.g. "直播中") at the top-right of the card."""
        font = self._resolver.resolve(tag[0], self.text_font, font_size)
        font.setSize(font_size)
        size = font.measureText(text=tag)  # type: ignore
        surface = skia.Surface(int(size + 20), int(font.getSize() + 20))
        canvas = surface.getCanvas()
        canvas.clear(skia.Color(*self.style.color.font_color.name_big_vip))
        blob = skia.TextBlob(text=tag, font=font)  # type: ignore
        paint = skia.Paint(AntiAlias=True, Color=skia.Color4f.kWhite)
        canvas.drawTextBlob(blob, 10, int(font.getSize() + 5), paint)
        tag_img = skia.Image.fromarray(
            array=canvas.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType),
            colorType=skia.ColorType.kRGBA_8888_ColorType,
        )  # type: ignore
        tag_img = await round_corners(tag_img, 10)
        await paste(self.canvas, tag_img, (1010 - tag_img.width(), 50))

    async def _make_sub_tag(self, text: str, font_size: int):
        """Draw a semi-transparent black overlay tag (e.g. duration)."""
        font = self._resolver.resolve(text[0], self.text_font, font_size)
        font.setSize(font_size)
        size = font.measureText(text=text)  # type: ignore
        surface = skia.Surface(int(size + 20), int(font.getSize() + 20))
        canvas = surface.getCanvas()
        canvas.clear(skia.Color(0, 0, 0, 150))
        blob = skia.TextBlob(text=text, font=font)  # type: ignore
        paint = skia.Paint(AntiAlias=True, Color=skia.Color4f.kWhite)
        canvas.drawTextBlob(blob, 10, int(font.getSize() + 5), paint)
        img = skia.Image.fromarray(
            array=canvas.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType),
            colorType=skia.ColorType.kRGBA_8888_ColorType,
        )  # type: ignore
        await paste(self.canvas, await round_corners(img, 10), (80, 525))

    def _bg(self, repost):
        return self.style.color.background.repost if repost else self.style.color.background.normal


# ---------------------------------------------------------------------------
# Concrete renderers
# ---------------------------------------------------------------------------


class DynMajorDraw:
    """Dynamic picture drawing (handled separately — not registered)."""

    def __init__(self, style: PolyStyle, items=None):
        self.style = style
        self.items = items

    async def run(self, repost: bool = False) -> Optional[np.ndarray]:
        try:
            item_count = len(self.items)
            bg = self.style.color.background.repost if repost else self.style.color.background.normal
            if item_count == 1:
                return await self._single_img(bg, self.items)
            elif item_count in {2, 4}:
                return await self._dual_img(bg, self.items)
            return await self._triplex_img(bg, self.items)
        except Exception as e:
            logger.exception(e)
            return None

    async def _single_img(self, bg, items):
        src = items[0].src or items[0].url
        img_height, img_width = items[0].height, items[0].width
        img_url = f"{src}@{600}w_{800}h_!header.webp" if img_height / img_width > 4 else src
        img = await fetch_images(img_url)
        if img is not None:
            img = img.resize(width=1008, height=int(img.height() * 1008 / img.width()))
            surface = skia.Surface(1080, img.height() + 20)
            canvas = surface.getCanvas()
            canvas.clear(skia.Color(*bg))
            await paste(canvas, img, (36, 10), clear_background=True)
        else:
            surface = skia.Surface(1080, 1080)
            canvas = surface.getCanvas()
            canvas.clear(skia.Color(*bg))
        return canvas.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)

    async def _dual_img(self, bg, items):
        url_list = []
        for item in items:
            src = item.src or item.url
            suffix = "@520w_520h_!header.webp" if item.height / item.width > 3 else "@520w_520h_1e_1c.webp"
            url_list.append(f"{src}{suffix}")
        imgs = await fetch_images(url_list, (520, 520))
        num = len(url_list) / 2
        back_size = int(num * 520 + 20 * num)
        surface = skia.Surface(1080, back_size)
        canvas = surface.getCanvas()
        canvas.clear(skia.Color(*bg))
        x, y = 15, 10
        for i in imgs:
            if i is not None:
                await paste(canvas, i, (x, y), clear_background=True)
            x += 530
            if x > 1000:
                x = 15
                y += 530
        return canvas.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)

    async def _triplex_img(self, bg, items):
        url_list = []
        for item in items:
            src = item.src or item.url
            suffix = "@260w_260h_!header.webp" if item.height / item.width > 3 else "@260w_260h_1e_1c.webp"
            url_list.append(f"{src}{suffix}")
        num = ceil(len(items) / 3)
        imgs = await fetch_images(url_list, (346, 346))
        back_size = int(num * 346 + 20 * num)
        surface = skia.Surface(1080, back_size)
        canvas = surface.getCanvas()
        canvas.clear(skia.Color(*bg))
        x, y = 11, 10
        for img in imgs:
            if img is not None:
                await paste(canvas, img, (x, y), clear_background=True)
            x += 356
            if x > 1000:
                x = 11
                y += 356
        return canvas.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)


# ---------------------------------------------------------------------------
# Registered major renderers
# ---------------------------------------------------------------------------


@register_major("MAJOR_TYPE_DRAW")
class MajorDraw(BaseMajorRenderer):
    async def run(self, repost: bool = False) -> Optional[np.ndarray]:
        return await DynMajorDraw(self.style, items=self.major.draw.items).run(repost)


@register_major("MAJOR_TYPE_ARCHIVE")
class MajorArchive(BaseMajorRenderer):
    async def run(self, repost: bool = False) -> Optional[np.ndarray]:
        bg = self._bg(repost)
        surface = skia.Surface(1080, 695)
        self.canvas = surface.getCanvas()
        self.canvas.clear(skia.Color(*bg))
        tv = skia.Image.open(path.join(self.src_path, "tv.png")).resize(130, 130)
        try:
            cover = await fetch_images(f"{self.major.archive.cover}@505w_285h_1c.webp", (1010, 570))
            await self._draw_shadow((35, 25, 1010, 655), 20, bg)
            rec = skia.Rect.MakeXYWH(35, 25, 1010, 665)
            self.canvas.clipRRect(skia.RRect(rec, 20, 20), skia.ClipOp.kIntersect)
            await self._draw_text(self.major.archive.title, self.style.font.font_size.text,
                                  (60, 650, 980, 600, 10), self.style.color.font_color.text)
            if cover is not None:
                await paste(self.canvas, cover, (35, 25))
            await paste(self.canvas, tv, (905, 455))
            if self.major.archive.badge is not None and self.major.archive.badge.text != "":
                await self._make_tag(self.major.archive.badge.text, self.style.font.font_size.text)
            await self._make_sub_tag(self.major.archive.duration_text, self.style.font.font_size.title)
            return self.canvas.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)
        except Exception as e:
            logger.exception(e)
            return None


@register_major("MAJOR_TYPE_LIVE_RCMD")
class MajorLiveRcmd(BaseMajorRenderer):
    async def run(self, repost: bool = False) -> Optional[np.ndarray]:
        bg = self._bg(repost)
        surface = skia.Surface(1080, 695)
        self.canvas = surface.getCanvas()
        self.canvas.clear(skia.Color(*bg))
        try:
            cover = await fetch_images(
                f"{self.major.live_rcmd.content.live_play_info.cover}@505w_285h_1c.webp", (1010, 570))
            await self._draw_shadow((35, 25, 1010, 655), 20, bg)
            rec = skia.Rect.MakeXYWH(35, 25, 1010, 665)
            self.canvas.clipRRect(skia.RRect(rec, 20, 20), skia.ClipOp.kIntersect)
            await self._draw_text(self.major.live_rcmd.content.live_play_info.title,
                                  self.style.font.font_size.text,
                                  (60, 650, 980, 600, 10), self.style.color.font_color.text)
            if cover is not None:
                await paste(self.canvas, cover, (35, 25))
            await self._make_tag("直播中", self.style.font.font_size.text)
            await self._make_sub_tag(
                self.major.live_rcmd.content.live_play_info.watched_show.text_large,
                self.style.font.font_size.title)
            return self.canvas.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)
        except Exception as e:
            logger.exception(e)
            return None


@register_major("MAJOR_TYPE_OPUS")
class MajorOpus(BaseMajorRenderer):
    @staticmethod
    def _convert_to_rich_text_detail(node: RichTextNodes) -> RichTextDetail:
        return RichTextDetail(
            type=node.type, text=node.text, orig_text=node.orig_text,
            emoji=node.emoji.dict() if node.type == "RICH_TEXT_NODE_TYPE_EMOJI" else None,
        )

    async def run(self, repost: bool = False) -> Optional[np.ndarray]:
        pics = []
        try:
            if self.major.opus.title:
                pics.append(await self._make_title(self.major.opus.title, repost))
        except Exception as e:
            logger.exception(e)
        try:
            if self.major.opus.summary:
                dyn_text = Text(
                    text=self.major.opus.summary.text, topic=None,
                    rich_text_nodes=[self._convert_to_rich_text_detail(n)
                                     for n in self.major.opus.summary.rich_text_nodes],
                )
                text_img = await BiliText(path.dirname(self.src_path), self.style).run(dyn_text, repost)
                pics.append(text_img)
        except Exception as e:
            logger.exception(e)
        try:
            if self.major.opus.pics:
                cover = await DynMajorDraw(self.style, items=self.major.opus.pics).run(repost)
                pics.append(cover)
        except Exception as e:
            logger.exception(e)
        if not pics:
            return None
        return await merge_pictures(pics)

    async def _make_title(self, title, repost):
        bg = self._bg(repost)
        surface = skia.Surface(1080, self.style.font.font_size.name + 20)
        canvas = surface.getCanvas()
        canvas.clear(skia.Color(*bg))
        await self._drawer.draw_text(
            canvas, title, self.style.font.font_size.name,
            (45, int((self.style.font.font_size.name + 40) / 2), 1035, 40, 0),
            self.style.color.font_color.text, font_style=skia.FontStyle().Bold(),
        )
        return canvas.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)


@register_major("MAJOR_TYPE_ARTICLE")
class MajorArticle(BaseMajorRenderer):
    async def run(self, repost: bool = False) -> Optional[np.ndarray]:
        bg = self._bg(repost)
        surface = skia.Surface(1080, 640)
        self.canvas = surface.getCanvas()
        self.canvas.clear(skia.Color(*bg))
        try:
            await self._draw_shadow((35, 20, 1010, 600), 20, bg)
            rec = skia.Rect.MakeXYWH(35, 20, 1010, 600)
            self.canvas.clipRRect(skia.RRect(rec, 20, 20), skia.ClipOp.kIntersect)
            await self._draw_title_and_desc()
            await self._make_cover()
            return self.canvas.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)
        except Exception as e:
            logger.exception(e)
            return None

    async def _make_cover(self):
        if len(self.major.article.covers) > 1:
            url_list = [f"{i}@360w_360h_1c" for i in self.major.article.covers]
            imgs = await fetch_images(url_list, (330, 330))
            for i, j in enumerate(imgs):
                if j is not None:
                    await paste(self.canvas, j, (35 + i * 340, 20))
        else:
            img = await fetch_images(f"{self.major.article.covers[0]}@647w_150h_1c.webp", (1010, 300))
            if img is not None:
                await paste(self.canvas, img, (35, 20))

    async def _draw_title_and_desc(self):
        title = self.major.article.title
        y_title = 410 if len(self.major.article.covers) > 1 else 390
        await self._draw_text(title, self.style.font.font_size.text,
                              (50, y_title, 960, 330, 0), self.style.color.font_color.text)
        await self._draw_text(self.major.article.desc, self.style.font.font_size.title,
                              (65, 460, 980, 620, int(self.style.font.font_size.title * 1.8)),
                              self.style.color.font_color.sub_title)


@register_major("MAJOR_TYPE_COMMON")
class MajorCommon(BaseMajorRenderer):
    async def run(self, repost: bool = False) -> Optional[np.ndarray]:
        bg = self._bg(repost)
        surface = skia.Surface(1080, 285)
        self.canvas = surface.getCanvas()
        self.canvas.clear(skia.Color(*bg))
        try:
            await self._draw_shadow((35, 20, 1010, 245), 20, bg)
            rec = skia.Rect.MakeXYWH(35, 20, 1010, 245)
            self.canvas.clipRRect(skia.RRect(rec, 20, 20), skia.ClipOp.kIntersect)
            cover = await fetch_images(f"{self.major.common.cover}@245w_245h_1c.webp", (245, 245))
            if cover is not None:
                await paste(self.canvas, cover, (35, 20))
            await self._make_title()
            await self._make_common_tag()
            return self.canvas.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)
        except Exception as e:
            logger.exception(e)
            return None

    async def _make_title(self):
        await self._draw_text(self.major.common.title, self.style.font.font_size.text,
                              (310, 120, 950, 120, 0), self.style.color.font_color.text)
        await self._draw_text(self.major.common.desc, self.style.font.font_size.title,
                              (310, 190, 970, 190, 0), self.style.color.font_color.sub_title)

    async def _make_common_tag(self):
        if self.major.common.badge is not None and self.major.common.badge.text != "":
            self.text_font.setSize(self.style.font.font_size.sub_title)
            size = self.text_font.measureText(self.major.common.badge.text)
            tag_width = int(size + 20)
            surface = skia.Surface(tag_width, int(self.text_font.getSize() + 20))
            canvas = surface.getCanvas()
            canvas.clear(skia.Color(*self.style.color.font_color.name_big_vip))
            blob = skia.TextBlob(self.major.common.badge.text, self.text_font)
            paint = skia.Paint(AntiAlias=True, Color=skia.Color4f.kWhite)
            canvas.drawTextBlob(blob, 10, int(self.text_font.getSize() + 5), paint)
            tag_img = skia.Image.fromarray(
                canvas.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType),
                colorType=skia.ColorType.kRGBA_8888_ColorType,
            )
            await paste(self.canvas, await round_corners(tag_img, 10), (280 - tag_width - 20, 40))


@register_major("MAJOR_TYPE_MUSIC")
class MajorMusic(BaseMajorRenderer):
    async def run(self, repost: bool = False) -> Optional[np.ndarray]:
        bg = self._bg(repost)
        surface = skia.Surface(1080, 285)
        self.canvas = surface.getCanvas()
        self.canvas.clear(skia.Color(*bg))
        try:
            await self._draw_shadow((35, 20, 1010, 245), 20, bg)
            rec = skia.Rect.MakeXYWH(35, 20, 1010, 245)
            self.canvas.clipRRect(skia.RRect(rec, 20, 20), skia.ClipOp.kIntersect)
            cover = await fetch_images(f"{self.major.music.cover}@245w_245h_1c.webp", (245, 245))
            if cover is not None:
                await paste(self.canvas, cover, (35, 20))
            await self._make_title()
            return self.canvas.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)
        except Exception as e:
            logger.exception(e)
            return None

    async def _make_title(self):
        await self._draw_text(self.major.music.title, self.style.font.font_size.text,
                              (310, 120, 950, 120, 0), self.style.color.font_color.text)
        await self._draw_text(self.major.music.label, self.style.font.font_size.title,
                              (310, 190, 970, 190, 0), self.style.color.font_color.sub_title)


@register_major("MAJOR_TYPE_PGC")
class MajorPgc(BaseMajorRenderer):
    async def run(self, repost: bool = False) -> Optional[np.ndarray]:
        bg = self._bg(repost)
        surface = skia.Surface(1080, 695)
        self.canvas = surface.getCanvas()
        self.canvas.clear(skia.Color(*bg))
        tv = skia.Image.open(path.join(self.src_path, "tv.png")).resize(130, 130)
        try:
            cover = await fetch_images(f"{self.major.pgc.cover}@505w_285h_1c.webp", (1010, 570))
            await self._draw_shadow((35, 25, 1010, 655), 20, bg)
            rec = skia.Rect.MakeXYWH(35, 25, 1010, 665)
            self.canvas.clipRRect(skia.RRect(rec, 20, 20), skia.ClipOp.kIntersect)
            await self._draw_text(self.major.pgc.title, self.style.font.font_size.text,
                                  (60, 650, 980, 600, 10), self.style.color.font_color.text)
            if cover is not None:
                await paste(self.canvas, cover, (35, 25))
            await paste(self.canvas, tv, (905, 455))
            tag = self.major.pgc.badge.text if (self.major.pgc.badge and self.major.pgc.badge.text) else "投稿视频"
            await self._make_tag(tag, self.style.font.font_size.text)
            await self._make_sub_tag(f"{self.major.pgc.stat.play}播放", self.style.font.font_size.title)
            return self.canvas.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)
        except Exception as e:
            logger.exception(e)
            return None


@register_major("MAJOR_TYPE_MEDIALIST")
class MajorMediaList(BaseMajorRenderer):
    async def run(self, repost: bool = False) -> Optional[np.ndarray]:
        bg = self._bg(repost)
        surface = skia.Surface(1080, 695)
        self.canvas = surface.getCanvas()
        self.canvas.clear(skia.Color(*bg))
        tv = skia.Image.open(path.join(self.src_path, "tv.png")).resize(130, 130)
        try:
            cover = await fetch_images(f"{self.major.medialist.cover}@505w_285h_1c.webp", (1010, 570))
            await self._draw_shadow((35, 25, 1010, 655), 20, bg)
            rec = skia.Rect.MakeXYWH(35, 25, 1010, 665)
            self.canvas.clipRRect(skia.RRect(rec, 20, 20), skia.ClipOp.kIntersect)
            await self._draw_text(self.major.medialist.title, self.style.font.font_size.text,
                                  (60, 650, 980, 600, 10), self.style.color.font_color.text)
            if cover is not None:
                await paste(self.canvas, cover, (35, 25))
            tag = self.major.medialist.badge.text if self.major.medialist.badge else "投稿视频"
            await self._make_tag(tag, self.style.font.font_size.text)
            await self._make_sub_tag(self.major.medialist.sub_title, self.style.font.font_size.title)
            await paste(self.canvas, tv, (905, 455))
            return self.canvas.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)
        except Exception as e:
            logger.exception(e)
            return None


@register_major("MAJOR_TYPE_COURSES")
class MajorCourses(BaseMajorRenderer):
    async def run(self, repost: bool = False) -> Optional[np.ndarray]:
        bg = self._bg(repost)
        surface = skia.Surface(1080, 695)
        self.canvas = surface.getCanvas()
        self.canvas.clear(skia.Color(*bg))
        tv = skia.Image.open(path.join(self.src_path, "tv.png")).resize(130, 130)
        try:
            cover = await fetch_images(f"{self.major.courses.cover}@505w_285h_1c.webp", (1010, 570))
            await self._draw_shadow((35, 25, 1010, 655), 20, bg)
            rec = skia.Rect.MakeXYWH(35, 25, 1010, 665)
            self.canvas.clipRRect(skia.RRect(rec, 20, 20), skia.ClipOp.kIntersect)
            await self._draw_text(self.major.courses.title, self.style.font.font_size.text,
                                  (60, 650, 980, 600, 10), self.style.color.font_color.text)
            if cover is not None:
                await paste(self.canvas, cover, (35, 25))
            tag = self.major.courses.badge.text if (self.major.courses.badge and self.major.courses.badge.text) else "投稿视频"
            await self._make_tag(tag, self.style.font.font_size.text)
            await self._make_sub_tag(self.major.courses.desc, self.style.font.font_size.title)
            await paste(self.canvas, tv, (905, 455))
            return self.canvas.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)
        except Exception as e:
            logger.exception(e)
            return None


@register_major("MAJOR_TYPE_UGC_SEASON")
class MajorUgc(BaseMajorRenderer):
    async def run(self, repost: bool = False) -> Optional[np.ndarray]:
        bg = self._bg(repost)
        surface = skia.Surface(1080, 695)
        self.canvas = surface.getCanvas()
        self.canvas.clear(skia.Color(*bg))
        tv = skia.Image.open(path.join(self.src_path, "tv.png")).resize(130, 130)
        try:
            cover = await fetch_images(f"{self.major.ugc_season.cover}@505w_285h_1c.webp", (1010, 570))
            await self._draw_shadow((35, 25, 1010, 655), 20, bg)
            rec = skia.Rect.MakeXYWH(35, 25, 1010, 665)
            self.canvas.clipRRect(skia.RRect(rec, 20, 20), skia.ClipOp.kIntersect)
            await self._draw_text(self.major.ugc_season.title, self.style.font.font_size.text,
                                  (60, 650, 980, 600, 10), self.style.color.font_color.text)
            if cover is not None:
                await paste(self.canvas, cover, (35, 25))
            tag = self.major.ugc_season.badge.text if (self.major.ugc_season.badge and self.major.ugc_season.badge.text) else "投稿视频"
            await self._make_tag(tag, self.style.font.font_size.text)
            await self._make_sub_tag(self.major.ugc_season.duration_text, self.style.font.font_size.title)
            await paste(self.canvas, tv, (905, 455))
            return self.canvas.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)
        except Exception as e:
            logger.exception(e)
            return None


@register_major("MAJOR_TYPE_LIVE")
class MajorLive(BaseMajorRenderer):
    async def run(self, repost: bool = False) -> Optional[np.ndarray]:
        bg = self._bg(repost)
        surface = skia.Surface(1080, 695)
        self.canvas = surface.getCanvas()
        self.canvas.clear(skia.Color(*bg))
        try:
            cover = await fetch_images(f"{self.major.live.cover}@505w_285h_1c.webp", (1010, 570))
            await self._draw_shadow((35, 25, 1010, 655), 20, bg)
            rec = skia.Rect.MakeXYWH(35, 25, 1010, 665)
            self.canvas.clipRRect(skia.RRect(rec, 20, 20), skia.ClipOp.kIntersect)
            await self._draw_text(self.major.live.title, self.style.font.font_size.text,
                                  (60, 650, 980, 600, 10), self.style.color.font_color.text)
            if cover is not None:
                await paste(self.canvas, cover, (35, 25))
            tag = self.major.live.badge.text if (self.major.live.badge and self.major.live.badge.text) else "投稿视频"
            await self._make_tag(tag, self.style.font.font_size.text)
            await self._make_sub_tag(self.major.live.desc_second, self.style.font.font_size.title)
            return self.canvas.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)
        except Exception as e:
            logger.exception(e)
            return None


@register_major("MAJOR_TYPE_NONE")
class MajorNone(BaseMajorRenderer):
    async def run(self, repost: bool = False) -> Optional[np.ndarray]:
        bg = self._bg(repost)
        surface = skia.Surface(1080, 100)
        self.canvas = surface.getCanvas()
        self.canvas.clear(skia.Color(*bg))
        try:
            error = skia.Image.open(path.join(self.src_path, "error.png")).resize(40, 40)
            await self._draw_text(self.major.none.tips, self.style.font.font_size.text,
                                  (90, 60, 1080, 40, 0), self.style.color.font_color.sub_title)
            await paste(self.canvas, error, (40, 30))
            return self.canvas.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)
        except Exception as e:
            logger.exception(e)
            return None


@register_major("MAJOR_TYPE_BLOCKED")
class MajorBlocked(BaseMajorRenderer):
    async def run(self, repost: bool = False) -> Optional[np.ndarray]:
        bg = self._bg(repost)
        surface = skia.Surface(1080, 1200)
        self.canvas = surface.getCanvas()
        self.canvas.clear(skia.Color(*bg))
        try:
            result = await fetch_images([
                f"{self.major.blocked.bg_img.img_dark}@1c.webp",
                self.major.blocked.icon.img_day,
            ])
            await self._draw_shadow((40, 100, 1000, 1000), 20, bg)
            rec = skia.Rect.MakeXYWH(40, 100, 1000, 1000)
            self.canvas.clipRRect(skia.RRect(rec, 20, 20), skia.ClipOp.kIntersect)
            if result[1] is not None:
                await paste(self.canvas, result[1], (456, 380))
            if result[0] is not None:
                await paste(self.canvas, result[0].resize(1000, 1000), (40, 100))
            text = self.major.blocked.hint_message.split("\n")
            await self._draw_text(text[0], self.style.font.font_size.name,
                                  (380, 630, 980, 600, 10), self.style.color.font_color.sub_title)
            await self._draw_text(text[1], self.style.font.font_size.name,
                                  (160, 700, 980, 600, 10), self.style.color.font_color.sub_title)
            return self.canvas.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)
        except Exception as e:
            logger.exception(e)
            return None
