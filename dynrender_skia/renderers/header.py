"""Header and repost-header renderers."""

import asyncio
from os import path
from time import localtime, strftime, time
from typing import Optional

import numpy as np
import skia
from dynamicadaptor.Header import Head
from loguru import logger

from ..config import PolyStyle
from ..graphics import TextDrawer, circle_crop, fetch_images, paste


class BiliHeader:
    """Render the main post header."""

    def __init__(self, static_path: str, style: PolyStyle) -> None:
        self.face_path = path.join(static_path, "Cache", "Face")
        self.pendant_path = path.join(static_path, "Cache", "Pendant")
        self.src_path = path.join(static_path, "Src")
        self.style = style
        self.canvas = None
        self.message = None
        self._drawer = TextDrawer(style)

    async def run(self, header_message: Head) -> Optional[np.ndarray]:
        try:
            self.message = header_message
            surface = skia.Surface(1080, 400)
            self.canvas = surface.getCanvas()
            self.canvas.clear(skia.Color(*self.style.color.background.normal))
            face_task = asyncio.ensure_future(self._get_face_and_pendant(True))
            pendant_task = asyncio.ensure_future(self._get_face_and_pendant())
            await self._paste_logo()
            await self._draw_name()
            await self._draw_pub_time()
            face = await face_task
            pendant = await pendant_task
            await self._past_face(face)
            await self._paste_pendant(pendant)
            await self._paste_vip()
            return self.canvas.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)
        except Exception as e:
            logger.exception(e)
            return None

    async def _paste_pendant(self, pendant: skia.Image) -> None:
        if pendant is not None:
            pendant = pendant.resize(190, 190)
            await paste(self.canvas, pendant, (10, 210))

    async def _paste_vip(self) -> None:
        if self.message.official_verify and self.message.official_verify.type != -1:
            if self.message.official_verify.type == 0:
                img_path = path.join(self.src_path, "official_yellow.png")
            else:
                img_path = path.join(self.src_path, "official_blue.png")
            img = skia.Image.open(img_path).resize(45, 45)
            await paste(self.canvas, img, (120, 330))
        elif self.message.vip and self.message.vip.status == 1:
            if self.message.vip.avatar_subscript == 1:
                img_path = path.join(self.src_path, "big_vip.png")
            else:
                img_path = path.join(self.src_path, "small_vip.png")
            img = skia.Image.open(img_path).resize(45, 45)
            await paste(self.canvas, img, (120, 330))

    async def _past_face(self, face: Optional[skia.Image]) -> None:
        if face:
            face = await circle_crop(face, 120)
            await paste(self.canvas, face, (45, 245))

    async def _get_face_and_pendant(self, img_type: bool = False) -> Optional[skia.Image]:
        if img_type:
            img_name = f"{self.message.mid}.webp"
            img_url = f"{self.message.face}@240w_240h_1c_1s.webp"
            img_path = path.join(self.face_path, img_name)
        elif self.message.pendant and self.message.pendant.image:
            img_name = f"{self.message.pendant.pid}.png"
            img_url = f"{self.message.pendant.image}@360w_360h.webp"
            img_path = path.join(self.pendant_path, img_name)
        else:
            return None
        if path.exists(img_path):
            if time() - int(path.getmtime(img_path)) <= 43200:
                return skia.Image.open(img_path)
        img = await fetch_images(img_url)
        if img is not None:
            img.save(img_path)
            return img
        return None

    async def _draw_pub_time(self) -> None:
        if self.message.pub_ts:
            pub_time = strftime("%Y-%m-%d %H:%M:%S", localtime(self.message.pub_ts))
        elif self.message.pub_time:
            pub_time = self.message.pub_time
        else:
            pub_time = " "
        await self._drawer.draw_text(
            self.canvas,
            pub_time,
            self.style.font.font_size.time,
            (200, 350, 1010, 350, 0),
            self.style.color.font_color.sub_title,
        )

    async def _paste_logo(self) -> None:
        logo = skia.Image.open(path.join(self.src_path, "bilibili.png")).resize(231, 105)
        await paste(self.canvas, logo, (433, 20))

    async def _draw_name(self) -> None:
        if self.message.vip and self.message.vip.status == 1:
            color = (
                self.style.color.font_color.name_big_vip
                if self.message.vip.avatar_subscript == 1
                else self.style.color.font_color.name_small_vip
            )
        else:
            color = self.style.color.font_color.text
        await self._drawer.draw_text(
            self.canvas,
            self.message.name,
            self.style.font.font_size.name,
            (200, 300, 1010, 300, 0),
            color,
        )


class RepostHeader:
    """Render the repost sub-header."""

    def __init__(self, static_path: str, style: PolyStyle) -> None:
        self.style = style
        self.static_path = static_path
        self._drawer = TextDrawer(style)

    async def run(self, message: Head) -> Optional[np.ndarray]:
        surface = skia.Surface(1080, 100)
        canvas = surface.getCanvas()
        canvas.clear(skia.Color(*self.style.color.background.repost))
        try:
            if not message.name:
                return None
            if message.face:
                pos = 140
                await self._draw_face(canvas, message.face, message.mid)
            else:
                pos = 35
            await self._draw_name(canvas, message.name, pos)
            return canvas.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)
        except Exception as e:
            logger.exception(e)
            return None

    async def _draw_face(self, canvas: skia.Canvas, url: str, mid: int) -> None:
        if url:
            img = await self._get_face(mid, url)
            if img is not None:
                face = await circle_crop(img, 80)
                await paste(canvas, face, (40, 10))

    async def _draw_name(self, canvas: skia.Canvas, name: str, pos: int) -> None:
        await self._drawer.draw_text(
            canvas, name, self.style.font.font_size.name,
            (pos, 70, 1010, 70, 0),
            self.style.color.font_color.rich_text,
        )

    async def _get_face(self, mid: int, url: str) -> Optional[skia.Image]:
        img_name = f"{mid}.webp"
        img_url = f"{url}@240w_240h_1c_1s.webp"
        img_path = path.join(self.static_path, "Cache", "Face", img_name)
        if path.exists(img_path):
            if time() - int(path.getmtime(img_path)) <= 43200:
                return skia.Image.open(img_path)
        img = await fetch_images(img_url)
        if img is not None:
            img.save(img_path)
        return img
