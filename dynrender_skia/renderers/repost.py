"""Repost composite renderer — assembles header + text + major + additional for forwards."""

import asyncio
from os import path
from typing import Optional

import numpy as np
from dynamicadaptor.Repost import Forward

from ..config import PolyStyle
from ..graphics import merge_pictures
from .header import RepostHeader
from .registry import get_additional_renderer, get_major_renderer
from .text import BiliText


class BiliRepost:
    """Render a forwarded post."""

    def __init__(self, static_path: str, style: PolyStyle) -> None:
        self.static_path = static_path
        self.style = style

    async def run(self, message: Forward) -> Optional[np.ndarray]:
        src_path = path.join(self.static_path, "Src")
        tasks = [RepostHeader(self.static_path, self.style).run(message.header)]
        if message.text is not None:
            tasks.append(BiliText(self.static_path, self.style).run(message.text, repost=True))
        if message.major is not None:
            cls = get_major_renderer(message.major.type)
            if cls:
                tasks.append(cls(src_path, self.style, message.major).run(True))
        if message.additional is not None:
            cls = get_additional_renderer(message.additional.type)
            if cls:
                tasks.append(cls(src_path, self.style, message.additional).run(True))
        result = await asyncio.gather(*tasks)
        return await merge_pictures(result)  # type: ignore
