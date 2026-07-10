"""Main rendering engine.

Orchestrates the compositing pipeline: a dynamic post is broken into
independent render tasks (header, text, major, forward, additional,
footer), executed concurrently via ``asyncio.gather``, then vertically
stacked into a single output image.

Design: **Composite + Pipeline** — each section is an independent
renderer producing a ``(height, 1080, 4)`` numpy array; the pipeline
merges them in order.
"""

import asyncio
from os import path
from typing import Optional

from dynamicadaptor.Message import RenderMessage
from loguru import logger

from .config import create_style, init_static_path
from .graphics import merge_pictures
from .renderers import (
    BiliHeader,
    BiliRepost,
    BiliText,
    Footer,
    get_additional_renderer,
    get_major_renderer,
)


class DynRender:
    """Entry point for rendering Bilibili dynamic content to images.

    Usage::

        render = DynRender(font_family="Noto Sans SC", static_path="/data")
        img_array = await render.run(message)

    Parameters:
        font_family: System font family name (defaults to "Noto Sans SC").
        emoji_font_family: Emoji font family (defaults to "Noto Color Emoji").
        font_style: One of ``Normal``, ``Bold``, ``Italic``, ``BoldItalic``.
        static_path: Absolute path to static assets directory.
    """

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
        """Render a complete dynamic post.

        The pipeline executes header → text → major → forward
        → additional → footer in parallel, then stacks the results
        vertically.

        Each section that is ``None`` in the message is skipped;
        unsupported major/additional types are logged as warnings
        but do not halt the render.
        """
        # Resource path shared by major/additional renderers
        src_path = path.join(self.static_path, "Src")

        # ---- build task list ----
        tasks = [BiliHeader(self.static_path, self.style).run(message.header)]

        if message.text is not None:
            tasks.append(BiliText(self.static_path, self.style).run(message.text))

        if message.major is not None:
            cls = get_major_renderer(message.major.type)
            if cls:
                tasks.append(cls(src_path, self.style, message.major).run())
            else:
                logger.warning(
                    f"{message.major.type} is not supported"
                )

        if message.forward is not None:
            tasks.append(
                BiliRepost(self.static_path, self.style).run(message.forward)
            )

        if message.additional is not None:
            cls = get_additional_renderer(message.additional.type)
            if cls:
                tasks.append(cls(src_path, self.style, message.additional).run())
            else:
                logger.warning(
                    f"{message.additional.type} IS NOT SUPPORT NOW"
                )

        tasks.append(Footer(self.static_path, self.style).run())

        # ---- execute and composite ----
        result = await asyncio.gather(*tasks)
        return await merge_pictures(result)
