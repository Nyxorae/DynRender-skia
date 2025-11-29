# @Time    : 2023/7/15 下午9:22
# @Author  : Polyisoprene
# @File    : DynTools.py
import asyncio
from typing import Optional, Union, cast

import emoji
import httpx
import numpy as np
import skia
from loguru import logger
from numpy import ndarray

from .DynStyle import PolyStyle
from .exception import ParseError


async def get_pictures(
    url: Union[str, list[str]], size: Optional[tuple[int, int]] = None, retries: int = 5
) -> Union[skia.Image, tuple[skia.Image, ...]]:
    """
    Asynchronously fetch images from a single URL or a list of URLs, optionally resizing them.

    This function creates an HTTP client with a specified retry policy, fetches images from
    the given URLs, and optionally resizes them to the specified dimensions using the skia library.

    Args:
        url (Union[str, list[str]]): A single URL or a list of URLs from which to fetch images.
        size (Optional[tuple[int, int]]): A tuple specifying the width and height to which the images should be
        resized. If None, images are returned in their original size.
        retries (int): The number of times to retry the request in case of connection issues. Default is 5 retries.

    Returns:
        Union[skia.Image, tuple[skia.Image, ...]]: A single skia.Image if a single URL is provided, or a tuple of
        skia.Images if a list of URLs is provided. If any request fails or an image cannot be decoded,
        the corresponding return value will be None.

    Raises:
        httpx.HTTPStatusError: If any HTTP request returns an unsuccessful status code.
        Exception: If an unexpected error occurs during the fetching or decoding of images.
    """
    transport = httpx.AsyncHTTPTransport(retries=retries)
    async with httpx.AsyncClient(transport=transport) as client:
        if isinstance(url, list):
            return await asyncio.gather(*[request_img(client, i, size) for i in url])
        else:
            return await request_img(client, url, size)


async def request_img(
    client: httpx.AsyncClient,
    url: str,
    size: Optional[tuple[int, int]],
) -> Optional[skia.Image]:
    """
    Fetch and optionally resize an image from a URL.

    This function performs an HTTP GET request using the provided client,
    decodes the response content into a ``skia.Image``, and optionally resizes
    the image if a target size is given.

    Network-related errors and non-2xx HTTP responses are propagated to the
    caller. If the image cannot be decoded from the response content, ``None``
    is returned.

    Args:
        client (httpx.AsyncClient):
            The HTTP client used to perform the request.
        url (str):
            The URL of the image to fetch.
        size (Optional[tuple[int, int]]):
            A ``(width, height)`` tuple specifying the desired output size.
            If ``None``, the image is returned at its original dimensions.

    Returns:
        Optional[skia.Image]:
            The decoded (and optionally resized) image, or ``None`` if the
            response content cannot be decoded into a valid image.

    Raises:
        httpx.RequestError:
            If a network error occurs while sending the request.
        httpx.HTTPStatusError:
            If the server responds with a non-2xx status code.
    """
    response = await client.get(url)
    response.raise_for_status()

    img = skia.Image.MakeFromEncoded(response.content)  # type: ignore[assignment]
    if img is None:
        logger.warning(f"Failed to decode image from {url}")
        return None

    if size is not None:
        width, height = size
        img = img.resize(width, height)

    return img


async def merge_pictures(img_list: list[ndarray]) -> ndarray:
    """
    Merge multiple images into a single image by stacking them vertically.

    This function takes a list of images as input and merges them into a single image by stacking them
    vertically. The width of each image must be 1080 pixels.

    Usage:
    ```python
    img1 = np.zeros([100, 1080, 4], np.uint8)
    img2 = np.zeros([200, 1080, 4], np.uint8)
    img3 = np.zeros([150, 1080, 4], np.uint8)
    img_list = [img1, img2, img3]
    merged_img = await merge_pictures(img_list)
    ```

    Args:
        img_list (list[ndarray]): A list of images to be merged.

    Returns:
        ndarray: A single image created by stacking the input images vertically.
    """
    img_top = np.zeros([0, 1080, 4], np.uint8)
    if len(img_list) == 1 and img_list[0] is not None:
        return img_list[0]
    for img in img_list:
        if img is None:
            continue
        if img.shape[1] != 1080:
            raise ValueError("The width of the image must be 1080")
        img_top = np.vstack((img_top, img))
    return img_top


async def paste(canvas: skia.Canvas, target: skia.Image, position: tuple, clear_background: bool = False) -> None:
    """
    Paste the target image onto the canvas at the specified position, with an option to clear the background.

    Args:
        canvas (skia.Canvas): The canvas on which the image will be drawn.
        target (skia.Image): The image to be pasted onto the canvas.
        position (tuple): A tuple (x, y) is the position on the canvas where the top-left corner of the image will be
        placed.
        clear_background (bool): If set to True, the background aera where the image will be placed is cleared to
        transparent before pasting the image. Defaults to False.

    Raises:
        ValueError: If the `target` image is None.
        AttributeError: If there is an issue accessing attributes or methods on `canvas` or `target`.

    Returns:
        None: The function does not return a value, but modifies the canvas in place.
    """
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


class DrawText:
    def __init__(self, style: PolyStyle):
        self.style = style
        self.text_font = skia.Font(
            skia.Typeface.MakeFromName(self.style.font.font_family, self.style.font.font_style),
            self.style.font.font_size.text,
        )
        self.emoji_font = skia.Font(
            skia.Typeface.MakeFromName(self.style.font.emoji_font_family, self.style.font.font_style),
            self.style.font.font_size.text,
        )

    @staticmethod
    def initialize_paint(font_color: tuple) -> skia.Paint:
        """
        Initialize a Paint object with the specified font color.
        """
        return skia.Paint(AntiAlias=True, Color=skia.Color(*font_color))

    @staticmethod
    def draw_ellipsis(canvas: skia.Canvas, x: int, y: int, font: skia.Font, paint: skia.Paint):
        """
        Draw an ellipsis at the specified position using the provided font and paint.
        """
        blob: skia.TextBlob = skia.TextBlob("...", font)  # type: ignore
        canvas.drawTextBlob(blob, x, y, paint)  # type: ignore

    def match_font(self, char: str, font_size: int) -> Optional[skia.Font]:
        """
        Matches a font for a given character and font size.

        This method uses the Skia Font Manager to find an appropriate font for the specified
        character based on the provided font size. It attempts to match a font that supports
        the character, considering the font family and style defined in the object's `style` attribute.

        Args:
            char (str): The character for which to find a matching font. Should be a single character string.
            font_size (int): The desired font size for the matched font.

        Returns:
            Optional[skia.Font]: A Skia `Font` object if a matching typeface is
            found, otherwise `None`.
        """
        if typeface := skia.FontMgr().matchFamilyStyleCharacter(
            self.style.font.font_family,
            self.style.font.font_style,
            ["zh", "en"],
            ord(char),
        ):
            return skia.Font(typeface, font_size)
        return None

    def set_font_sizes(self, font_size: int):
        """
        Set the font sizes for the text and emoji fonts.
        """
        self.text_font.setSize(font_size)
        self.emoji_font.setSize(font_size)

    async def extract_emoji_info(self, text: str) -> tuple[str, dict[int, list[Union[int, str]]]]:
        """
        Removes tabs from the given text and extracts emoji information.

        This function removes all tab characters from the provided text, then extracts
        and processes information about any emojis present in the text. It returns
        the cleaned text and a dictionary containing the emoji information.

        Args:
            text (str): The input text containing possible emojis.

        Usage:
        ```python
        text = "Hello, 🌍!"
        cleaned_text, emoji_info = await extract_emoji_info(text)
        print(cleaned_text)
        print(emoji_info)
        # Output: "Hello, 🌍!", {7: [8, '🌍']}
        ```

        Returns:
            tuple: A tuple containing:
                - The cleaned text without tab characters (str).
                - A dictionary with keys as integer positions of emojis in the text,
                  and values as lists containing the emoji index and the corresponding
                  emoji string (dict[int, list[Union[int, str]]]).
        """
        text = text.replace("\t", "")
        emoji_info: dict[int, list[Union[int, str]]] = await self.get_emoji_text(text)
        return text, emoji_info

    @staticmethod
    async def get_emoji_text(text: str) -> dict[int, list[Union[int, str]]]:
        """
        Get the positions of emojis in the text and their corresponding Unicode characters.

        Args:
            text (str): The text in which to search for emojis.

        Usage:
        ```python
        text = "Hello, 🌍!"
        emoji_info = await get_emoji_text(text)
        print(emoji_info)
        # Output: {7: [8, '🌍']}
        ```

        Returns:
            dict[int, list[Union[int, str]]]: A dictionary with the starting position of each emoji in the text as the key,
            and a list containing the end position of the emoji and the emoji character as the value.
        """
        result = emoji.emoji_list(text)
        return {i["match_start"]: [i["match_end"], i["emoji"]] for i in result}

    async def draw_text(
        self,
        canvas: skia.Canvas,
        text: str,
        font_size: int,
        position_and_bounds: tuple[int, int, int, int, int],
        font_color: tuple,
    ):
        self.set_font_sizes(font_size)
        paint = self.initialize_paint(font_color)

        text, emoji_info = await self.extract_emoji_info(text)
        text_length = len(text) - 1
        current_x, current_y, max_width, max_height, line_spacing = position_and_bounds
        initial_x = current_x  # 保存初始 x 坐标
        offset: int = 0

        while offset <= text_length:
            character = text[offset]

            if character == "\n":
                break

            if offset in emoji_info:
                offset, character, font = self._handle_emoji(offset, emoji_info)
            else:
                offset += 1
                font = self.text_font

            if not self._font_contains_character(font, character):
                font = self.match_font(character, font_size) or self.text_font  # pragma: no cover

            measure = font.measureText(character)
            blob = skia.TextBlob(character, font)
            canvas.drawTextBlob(blob, current_x, current_y, paint)
            current_x += measure

            if self._needs_new_line(current_x, max_width):
                current_y, current_x = self._advance_to_next_line(
                    current_y, line_spacing, max_height, initial_x, canvas, font, paint, current_x
                )
                if current_y >= max_height:
                    break

    def _handle_emoji(self, offset: int, emoji_info: dict[int, list[Union[int, str]]]) -> tuple[int, str, skia.Font]:
        try:
            character = cast(str, emoji_info[offset][1])
            end_pos = cast(int, emoji_info[offset][0])
            offset = end_pos
            font = self.emoji_font
            return offset, character, font
        except KeyError as e:
            raise ParseError(f"Error parsing emoji information {e}") from e

    @staticmethod
    def _font_contains_character(font: skia.Font, character: str) -> bool:
        return font.textToGlyphs(character)[0] != 0

    @staticmethod
    def _needs_new_line(current_x: int, max_width: int) -> bool:
        return current_x > max_width

    def _advance_to_next_line(
        self,
        current_y: int,
        line_spacing: int,
        max_height: int,
        initial_x: int,
        canvas: skia.Canvas,
        font: skia.Font,
        paint: skia.Paint,
        current_x: int,
    ) -> tuple[int, int]:
        if current_y + line_spacing >= max_height:
            self.draw_ellipsis(canvas, current_x, current_y, font, paint)
            return max_height, initial_x

        current_y += line_spacing
        return current_y, initial_x
