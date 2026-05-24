"""Dynamic text renderer — handles rich text with emoji, topics, and mixed content."""

import asyncio
from os import path
from typing import Optional

import emoji
import numpy as np
import skia
from dynamicadaptor.Content import Text

from ..config import PolyStyle
from ..graphics import TextDrawer, fetch_images, paste, merge_pictures
from ..typesetter import Atom, CharClass, KnuthPlassLineBreaker, classify_char


class BiliText:
    """Render the text portion of a dynamic post."""

    def __init__(self, static_path: str, style: PolyStyle) -> None:
        self.emoji_path = path.join(static_path, "Cache", "Emoji")
        self.src_path = path.join(static_path, "Src")
        self.style = style
        surface = skia.Surface(1080, 60)
        self.canvas = surface.getCanvas()
        self.bg_color = None
        self.x_bound = 1030
        self.start_x = 40
        self.line_spacing = int(self.style.font.font_size.text * 1.5)
        self.image_list = []
        self.emoji_dict = {}
        self._drawer = TextDrawer(style)

    async def run(self, dyn_text: Text, repost: bool = False) -> Optional[np.ndarray]:
        self.text_font = skia.Font(
            skia.Typeface.MakeFromName(self.style.font.font_family, self.style.font.font_style),
            self.style.font.font_size.text,
        )
        self.emoji_font = skia.Font(
            skia.Typeface.MakeFromName(self.style.font.emoji_font_family, self.style.font.font_style),
            self.style.font.font_size.text,
        )
        self.bg_color = self.style.color.background.repost if repost else self.style.color.background.normal
        self.canvas.clear(skia.Color(*self.bg_color))
        try:
            tasks = []
            if dyn_text.topic is not None:
                tasks.append(self._make_topic(dyn_text.topic.name))
            if dyn_text.text:
                tasks.append(self._make_text_image(dyn_text))
            await asyncio.gather(*tasks)
            return await merge_pictures(self.image_list)
        except Exception:
            return None

    async def _make_text_image(self, dyn_text):
        emoji_list = []
        emoji_name_list = []
        rich_list = []
        for i in dyn_text.rich_text_nodes:
            if i.type == "RICH_TEXT_NODE_TYPE_EMOJI":
                if i.text not in emoji_name_list:
                    emoji_name_list.append(i.text)
                    emoji_list.append(i.emoji.icon_url)
            elif i.type != "RICH_TEXT_NODE_TYPE_TEXT":
                rich_list.append(i)
        result = await asyncio.gather(
            self._get_emoji(emoji_list, emoji_name_list),
            self._get_rich_pics(rich_list),
        )
        await self._render_with_typesetter(result[1], dyn_text)

    async def _render_with_typesetter(self, rich_list: dict, dyn_text: Text):
        """Collect atoms from all richtext nodes, line-break, then render."""
        atoms = self._collect_atoms(dyn_text, rich_list)
        if not atoms:
            return

        max_w = self.x_bound - self.start_x
        fs = self.style.font.font_size.text
        stretch_sp = fs * 0.25
        shrink_sp = fs * 0.125
        breaker = KnuthPlassLineBreaker(
            max_width=max_w, indent=0,
            stretch_spacing=stretch_sp, shrink_spacing=shrink_sp,
        )
        lines = breaker.break_lines(atoms)

        for line_idx, (si, ei, ratio) in enumerate(lines):
            surface = skia.Surface(1080, self.line_spacing + 10)
            canvas = surface.getCanvas()
            canvas.clear(skia.Color(*self.bg_color))
            x = self.start_x
            n_atoms = ei - si
            extra_per_gap = 0.0
            if ratio > 0 and n_atoms > 1:
                extra_per_gap = ratio * stretch_sp
            elif ratio < 0 and n_atoms > 1:
                extra_per_gap = ratio * shrink_sp
            for k in range(si, ei):
                atom = atoms[k]
                if atom.char_class == CharClass.MANDATORY_BREAK:
                    continue
                if atom.icon_image is not None:
                    icon = atom.icon_image
                    await paste(canvas, icon, (int(x), int(60 - icon.dimensions().height()) / 2))
                    x += atom.width + extra_per_gap
                    continue
                color = atom.paint_color or skia.Color(*self.style.color.font_color.text)
                paint = skia.Paint(AntiAlias=True, Color=color)
                font = atom.font or self.text_font
                if font.textToGlyphs(atom.text)[0] == 0:
                    if typeface := skia.FontMgr().matchFamilyStyleCharacter(
                        self.style.font.font_family, self.style.font.font_style,
                        ["zh", "en"], ord(atom.text[0]),
                    ):
                        font = skia.Font(typeface, self.style.font.font_size.text)
                    else:
                        font = self.text_font
                draw_text = atom.text
                if atom.char_class == CharClass.EMOJI and len(atom.text) > 1:
                    if any(g == 0 for g in font.textToGlyphs(atom.text)):
                        # Font can't render the full sequence — strip modifiers
                        draw_text = atom.text[0]
                    else:
                        # HarfBuzz shaping for compound emoji (skin tones,
                        # variation selectors).  y=5 matches the old top-aligned
                        # emoji baseline; MakeFromShapedText uses a different
                        # coordinate convention than plain TextBlob.
                        blob = skia.TextBlob.MakeFromShapedText(atom.text, font)
                        canvas.drawTextBlob(blob, x, 5, paint)
                        x += atom.width + extra_per_gap
                        continue
                blob = skia.TextBlob(draw_text, font)
                canvas.drawTextBlob(
                    blob, x,
                    int(60 - (60 - self.style.font.font_size.text) / 2),
                    paint,
                )
                x += atom.width + extra_per_gap
            self.image_list.append(canvas.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType))

    def _collect_atoms(self, dyn_text: Text, rich_list: dict) -> list[Atom]:
        """Convert all richtext nodes into a flat list of Atoms for line-breaking.

        Uses inline font and width caches to avoid repeated ``measureText``
        and ``matchFamilyStyleCharacter`` calls for frequently-used characters.
        """
        atoms: list[Atom] = []
        # Local caches (hot path — called once per render, many chars)
        font_cache: dict[tuple[str, int], skia.Font] = {}
        width_cache: dict[tuple[str, int, int], float] = {}
        text_font_id = self.text_font.getTypeface().uniqueID()

        for node in dyn_text.rich_text_nodes:
            node_type = node.type
            is_rich = node_type in {"RICH_TEXT_NODE_TYPE_AT", "RICH_TEXT_NODE_TYPE_TOPIC"}
            is_text = node_type == "RICH_TEXT_NODE_TYPE_TEXT"

            if is_rich or is_text:
                color = (
                    skia.Color(*self.style.color.font_color.rich_text) if is_rich
                    else skia.Color(*self.style.color.font_color.text)
                )
                text = node.text.translate(str.maketrans({"\r": "", "\t": ""}))
                emoji_info = emoji.emoji_list(text)
                emoji_map = {e["match_start"]: [e["match_end"], e["emoji"]] for e in emoji_info}
                total = len(text)
                offset = 0
                while offset < total:
                    ch = text[offset]
                    if ch == "\n":
                        atoms.append(Atom("\n", 0.0, CharClass.MANDATORY_BREAK, paint_color=color))
                        offset += 1
                        continue
                    if offset in emoji_map:
                        end_pos, emoji_char = emoji_map[offset]
                        offset = end_pos
                        # Zero-width emoji modifiers — keep in atom for
                        # correct combined rendering, measure base only.
                        if offset < total and text[offset] in '️︎':
                            offset += 1
                        font = self.emoji_font
                        ch = emoji_char
                        render_ch = ch  # full emoji for rendering
                        while len(ch) > 1 and (ch[-1] in '️︎'
                                               or '\U0001f3fb' <= ch[-1] <= '\U0001f3ff'):
                            ch = ch[:-1]
                        # Fall back to system font if emoji font can't render it
                        if font.textToGlyphs(ch)[0] == 0:
                            if typeface := skia.FontMgr().matchFamilyStyleCharacter(
                                "", skia.FontStyle().Normal(), [], ord(ch[0]),
                            ):
                                font = skia.Font(typeface, self.style.font.font_size.text)
                        w = font.measureText(ch)
                        char_class = CharClass.EMOJI
                        # If font can't render the full emoji sequence
                        # (modifiers = glyph 0), fall back to base char
                        # to avoid tofu squares.
                        if len(render_ch) > 1 and any(g == 0 for g in font.textToGlyphs(render_ch)):
                            render_ch = ch
                        atoms.append(Atom(render_ch, w, char_class, font, paint_color=color))
                    else:
                        offset += 1
                        # Try width cache first
                        wkey = (ch, self.style.font.font_size.text, text_font_id)
                        if wkey in width_cache:
                            w = width_cache[wkey]
                            char_class = classify_char(ch)
                            font = self.text_font
                        else:
                            font = self.text_font
                            if font.textToGlyphs(ch)[0] == 0:
                                fkey = (ch, self.style.font.font_size.text)
                                if fkey in font_cache:
                                    font = font_cache[fkey]
                                elif typeface := skia.FontMgr().matchFamilyStyleCharacter(
                                    self.style.font.font_family,
                                    self.style.font.font_style,
                                    ["zh", "en"], ord(ch[0]),
                                ):
                                    font = skia.Font(typeface, self.style.font.font_size.text)
                                    font_cache[fkey] = font
                            w = font.measureText(ch)
                            char_class = classify_char(ch)
                            width_cache[wkey] = w
                        atoms.append(Atom(ch, w, char_class, font, paint_color=color))

            elif node_type == "RICH_TEXT_NODE_TYPE_EMOJI":
                img = self.emoji_dict.get(node.text)
                if img is not None:
                    w = img.dimensions().width() + 5
                    atoms.append(Atom(node.text, w, CharClass.EMOJI, icon_image=img))

            else:
                # Rich icon (vote, lottery, goods, link, cv)
                key_map = {
                    "RICH_TEXT_NODE_TYPE_VOTE": "vote",
                    "RICH_TEXT_NODE_TYPE_LOTTERY": "lottery",
                    "RICH_TEXT_NODE_TYPE_GOODS": "goods",
                    "RICH_TEXT_NODE_TYPE_CV": "cv",
                }
                key = key_map.get(node_type, "link")
                icon = rich_list.get(key)
                color = skia.Color(*self.style.color.font_color.rich_text)
                if icon is not None:
                    w = icon.dimensions().width() + 5
                    atoms.append(Atom(f"[{key}]", w, CharClass.INLINE_OBJECT,
                                      icon_image=icon, paint_color=color))
                # Link text after icon (cached measurement)
                for ch in node.text:
                    wkey = (ch, self.style.font.font_size.text, text_font_id)
                    if wkey in width_cache:
                        w = width_cache[wkey]
                        font = self.text_font
                    else:
                        font = self.text_font
                        if font.textToGlyphs(ch)[0] == 0:
                            fkey = (ch, self.style.font.font_size.text)
                            if fkey in font_cache:
                                font = font_cache[fkey]
                            elif typeface := skia.FontMgr().matchFamilyStyleCharacter(
                                self.style.font.font_family,
                                self.style.font.font_style,
                                ["zh", "en"], ord(ch),
                            ):
                                font = skia.Font(typeface, self.style.font.font_size.text)
                                font_cache[fkey] = font
                        w = font.measureText(ch)
                        width_cache[wkey] = w
                    atoms.append(Atom(ch, w, classify_char(ch), font, paint_color=color))

        return atoms

    async def _get_emoji(self, emoji_urls: list, emoji_names: list):
        emoji_pics = []
        emoji_index = []
        emoji_urls_to_fetch = []
        icon_size = int(self.style.font.font_size.text * 1.5)
        for i, emoji_text in enumerate(emoji_names):
            p = path.join(self.emoji_path, f"{emoji_text}.png")
            if path.exists(p):
                emoji_pics.append(skia.Image.open(p))
            else:
                emoji_urls_to_fetch.append(emoji_urls[i])
                emoji_index.append(i)
        if emoji_urls_to_fetch:
            result = await fetch_images(emoji_urls_to_fetch, (icon_size, icon_size))
            for i, j in enumerate(emoji_index):
                emoji_path = path.join(self.emoji_path, f"{emoji_names[j]}.png")
                emoji_pics.insert(j, result[i])
                if result[i] is not None:
                    result[i].save(emoji_path)
        self.emoji_dict = {name: emoji_pics[i] for i, name in enumerate(emoji_names)}

    async def _get_rich_pics(self, rich_list):
        """Return a dict of icon images for the given rich-text nodes.

        Icons are loaded once and cached on the instance (hot path —
        the same icons appear in many posts).
        """
        if not hasattr(self, '_icon_cache'):
            self._icon_cache: dict[str, skia.Image] = {}

        rich_dic = {}
        rich_size = self.style.font.font_size.text
        icon_map = {
            "RICH_TEXT_NODE_TYPE_VOTE": "vote.png",
            "RICH_TEXT_NODE_TYPE_LOTTERY": "lottery.png",
            "RICH_TEXT_NODE_TYPE_GOODS": "taobao.png",
            "RICH_TEXT_NODE_TYPE_WEB": "link.png",
            "RICH_TEXT_NODE_TYPE_BV": "link.png",
            "RICH_TEXT_NODE_TYPE_CV": "article.png",
        }
        for i in rich_list:
            icon_name = icon_map.get(i.type, "link.png")
            key = icon_name.split(".")[0]
            if key == "taobao":
                key = "goods"
            if key not in rich_dic:
                if key in self._icon_cache:
                    rich_dic[key] = self._icon_cache[key]
                else:
                    img = skia.Image.open(path.join(self.src_path, icon_name))
                    img = img.resize(rich_size, rich_size)
                    self._icon_cache[key] = img
                    rich_dic[key] = img
        return rich_dic

    async def _make_topic(self, topic: str) -> None:
        topic_size = self.style.font.font_size.text
        topic_img = skia.Image.open(path.join(self.src_path, "new_topic.png")).resize(topic_size, topic_size)
        icon_size = int(topic_size * 1.5)
        surface = skia.Surface(1080, icon_size + 10)
        canvas = surface.getCanvas()
        canvas.clear(skia.Color(*self.bg_color))
        await paste(canvas, topic_img, (45, 15))
        await self._drawer.draw_text(
            canvas, topic, topic_size,
            (45 + topic_size + 10, 50, 1080, 50, 0),
            self.style.color.font_color.rich_text,
        )
        self.image_list.append(canvas.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType))
