"""Main rendering engine — orchestrates the compositing pipeline."""

import asyncio
from os import path
from typing import Optional

from dynamicadaptor.Message import RenderMessage

from .config import create_style, init_static_path
from .graphics import merge_pictures
from .renderers import BiliHeader, Footer, BiliText, BiliRepost, get_major_renderer, get_additional_renderer


class DynRender:
    """Entry point for rendering Bilibili dynamic content to images."""

    def __init__(
        self,
        font_family: str = "Noto Sans SC",
        emoji_font_family: str = "Noto Color Emoji",
        font_style: str = "Normal",
        static_path: Optional[str] = None,
    ) -> None:
        self.static_path = init_static_path(static_path)
        self.style = create_style(font_family, emoji_font_family, font_style)

    async def run(self, message: RenderMessage):
        src_path = path.join(self.static_path, "Src")
        tasks = [BiliHeader(self.static_path, self.style).run(message.header)]

        if message.text is not None:
            tasks.append(BiliText(self.static_path, self.style).run(message.text))

        if message.major is not None:
            cls = get_major_renderer(message.major.type)
            if cls:
                tasks.append(cls(src_path, self.style, message.major).run())
            else:
                import logging
                logging.getLogger().warning(f"{message.major.type} is not supported")

        if message.forward is not None:
            tasks.append(BiliRepost(self.static_path, self.style).run(message.forward))

        if message.additional is not None:
            cls = get_additional_renderer(message.additional.type)
            if cls:
                tasks.append(cls(src_path, self.style, message.additional).run())
            else:
                import logging
                logging.getLogger().warning(f"{message.additional.type} IS NOT SUPPORT NOW")

        tasks.append(Footer(self.static_path, self.style).run())
        result = await asyncio.gather(*tasks)
        return await merge_pictures(result)
