import pathlib
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import httpx
import numpy as np
import pytest
import pytest_asyncio
import respx
import skia

from dynrender_skia.DynConfig import SetDynStyle
from dynrender_skia.DynTools import merge_pictures, request_img, get_pictures, DrawText, paste
from dynrender_skia.exception import ParseError


@pytest.mark.asyncio
class TestMergePictures:
    async def test_with_single_valid_image(self) -> None:
        img_list = [np.zeros([1080, 1080, 4], np.uint8)]
        result = await merge_pictures(img_list)
        assert np.array_equal(result, img_list[0])

    async def test_with_multiple_valid_images(self) -> None:
        img1 = np.zeros([1080, 1080, 4], np.uint8)
        img2 = np.ones([1080, 1080, 4], np.uint8)
        img_list = [img1, img2]
        result = await merge_pictures(img_list)
        expected = np.vstack((img1, img2))
        assert np.array_equal(result, expected)

    async def test_with_some_invalid_images(self) -> None:
        img1 = np.zeros([1080, 1920, 4], np.uint8)
        img2 = None
        img_list = [img1, img2]
        with pytest.raises(ValueError, match="The width of the image must be 1080") as exc_info:
            await merge_pictures(img_list)
        assert "The width of the image must be 1080" in str(exc_info.value)

    async def test_with_all_invalid_images(self) -> None:
        img_list = [None, None]
        result = await merge_pictures(img_list)  # type: ignore
        expected = np.zeros([0, 1080, 4], np.uint8)
        assert np.array_equal(result, expected)

    async def test_with_empty_list(self) -> None:
        img_list = []
        result = await merge_pictures(img_list)
        expected = np.zeros([0, 1080, 4], np.uint8)
        assert np.array_equal(result, expected)


@pytest.mark.asyncio
async def test_request_img_with_respx(resource_dir: pathlib.Path) -> None:
    with respx.mock() as mock:
        url = "http://bilibili.com"

        img_path = Path(resource_dir / "bilibili.png")
        img_content = img_path.read_bytes()

        route = mock.get(url)
        route.respond(content=img_content, status_code=200)

        async with httpx.AsyncClient() as client:
            size = (100, 100)
            img = await request_img(client, url, size)

            if img is None:
                return  # pragma: no cover
            assert img.height() == 100
            assert img.width() == 100


@pytest.mark.asyncio
async def test_request_img_with_respx_invalid_url() -> None:
    async with respx.mock() as mock:
        url = "http://bilibili.com"

        mock.get(url).mock(side_effect=httpx.ConnectError("Connection error"))

        async with httpx.AsyncClient() as client:
            with pytest.raises(httpx.ConnectError):
                await request_img(client, url, None)


@pytest.mark.asyncio
class TestRequestImg:
    @pytest_asyncio.fixture(scope="session")
    async def client(self):
        async with httpx.AsyncClient() as client_instance:
            yield client_instance

    async def test_request_img_success(
        self, client: httpx.AsyncClient, mock_img_url: str, img_path: pathlib.Path
    ) -> None:
        async with respx.mock() as mock:
            img_content = img_path.read_bytes()
            mock.get(mock_img_url).respond(content=img_content, status_code=200)
            size = (100, 100)
            img: skia.Image = await request_img(client, mock_img_url, size)
            assert img.height() == 100
            assert img.width() == 100

    async def test_request_img_with_http_exception(self, client: httpx.AsyncClient, mock_img_url: str) -> None:
        async with respx.mock(base_url=mock_img_url) as mock:
            mock.get(mock_img_url).mock(side_effect=httpx.ConnectError("Connection error"))

            with pytest.raises(httpx.ConnectError):
                await request_img(client, mock_img_url, None)

    async def test_skia_encode_exception(self, client: httpx.AsyncClient, mock_img_url: str, caplog) -> None:
        async with respx.mock() as mock:
            img_content = None
            mock.get(mock_img_url).respond(content=img_content, status_code=200)
            result = await request_img(client, mock_img_url, None)
            assert result is None
            assert "Failed to decode image" in caplog.text

    async def test_unexpected_exception(self, client: httpx.AsyncClient, mock_img_url: str):
        async with respx.mock(base_url=mock_img_url) as mock:
            mock.get(mock_img_url).mock(side_effect=TypeError("Invalid type provided"))

            with pytest.raises(TypeError):
                await request_img(client, mock_img_url, None)


@pytest.mark.asyncio
class TestGetPictures:
    @pytest.fixture
    def mock_skia_image(self, img_path: Path):
        return skia.Image.MakeFromEncoded(encoded=img_path.read_bytes())  # type: ignore

    async def test_get_pictures_with_single_url(
        self, mock_img_url: str, img_path: pathlib.Path, mock_skia_image: skia.Image
    ) -> None:
        async with respx.mock(base_url=mock_img_url) as mock:
            img_content = img_path.read_bytes()
            mock.get(mock_img_url).respond(content=img_content, status_code=200)
            image: skia.Image = await get_pictures(mock_img_url, None)

            img_array = image.tobytes()
            result = mock_skia_image.tobytes()
            assert img_array == result

    async def test_get_pictures_with_multiple_urls(self, mock_img_url: str, img_path: pathlib.Path) -> None:
        async with respx.mock(base_url=mock_img_url) as mock:
            img_content = img_path.read_bytes()
            mock.get(mock_img_url).respond(content=img_content, status_code=200)

            img_list = await get_pictures([mock_img_url, mock_img_url], None)
            assert len(img_list) == 2
            assert all(img is not None for img in img_list)

    async def test_initialize_paint(self):
        font_color = (255, 0, 0, 255)

        paint = DrawText.initialize_paint(font_color)

        assert isinstance(paint, skia.Paint)
        assert paint.isAntiAlias() is True, "Paint should enabled anti-alias"
        assert paint.getColor() == skia.Color(*font_color), "Paint color should be the same as font color"

    async def test_draw_ellipsis(self, mocker):
        x, y = 100, 50
        font = skia.Font(skia.Typeface("Arial"), 20)
        font_color = (0, 0, 0, 255)
        paint = DrawText.initialize_paint(font_color)

        canvas = mocker.MagicMock(spec=skia.Canvas)
        mock_text_blob = mocker.patch("skia.TextBlob")

        DrawText.draw_ellipsis(canvas, x, y, font, paint)

        mock_text_blob.assert_called_once_with("...", font)
        canvas.drawTextBlob.assert_called_once_with(mock_text_blob.return_value, x, y, paint)


@pytest.mark.asyncio
class TestDrawText:
    @pytest_asyncio.fixture(autouse=True)
    async def setup(self):
        font_family = "Noto Sans SC"
        emoji_font_family = "Noto Color Emoji"
        font_style = "Normal"
        self.style = SetDynStyle(font_family, emoji_font_family, font_style).set_style

    async def test_extract_emoji_info_removes_tabs_and_extracts_emoji(self):
        draw_text = DrawText(style=self.style)
        text = "Hello,\t🌍!"
        cleaned_text, emoji_info = await draw_text.extract_emoji_info(text)
        assert cleaned_text == "Hello,🌍!"
        assert emoji_info == {6: [7, "🌍"]}

    async def test_extract_emoji_info_handles_no_emoji(self):
        draw_text = DrawText(style=self.style)
        text = "Hello, World!"
        cleaned_text, emoji_info = await draw_text.extract_emoji_info(text)
        assert cleaned_text == "Hello, World!"
        assert emoji_info == {}

    async def test_extract_emoji_info_handles_multiple_emojis(self):
        draw_text = DrawText(style=self.style)
        text = "Hello, 🌍! How are you? 😊"
        cleaned_text, emoji_info = await draw_text.extract_emoji_info(text)
        assert cleaned_text == "Hello, 🌍! How are you? 😊"
        assert emoji_info == {7: [8, "🌍"], 23: [24, "😊"]}

    async def test_extract_emoji_info_handles_only_tabs(self):
        draw_text = DrawText(style=self.style)
        text = "\t\t\t"
        cleaned_text, emoji_info = await draw_text.extract_emoji_info(text)
        assert cleaned_text == ""
        assert emoji_info == {}

    async def test_needs_new_line_returns_true_when_exceeding_max_width(self):
        assert DrawText._needs_new_line(1100, 1080) is True

    async def test_needs_new_line_returns_false_when_within_max_width(self):
        assert DrawText._needs_new_line(900, 1080) is False

    async def test_needs_new_line_returns_true_when_equal_to_max_width(self):
        assert DrawText._needs_new_line(1080, 1080) is False

    async def test_font_contains_character_returns_true_for_valid_character(self):
        font = skia.Font(skia.Typeface.MakeDefault(), 12)  # type: ignore
        assert DrawText._font_contains_character(font, "A") is True

    async def test_font_contains_character_returns_false_for_invalid_character(self):
        font = skia.Font(skia.Typeface.MakeDefault(), 12)  # type: ignore
        assert DrawText._font_contains_character(font, "\uffff") is False

    async def test_advance_to_next_line_adds_line_spacing_when_within_max_height(self):
        draw_text = DrawText(style=self.style)
        canvas = skia.Surface(1080, 1920).getCanvas()
        current_y, current_x = draw_text._advance_to_next_line(
            current_y=100,
            line_spacing=20,
            max_height=200,
            initial_x=0,
            canvas=canvas,
            font=skia.Font(),
            paint=skia.Paint(),
            current_x=50,
        )
        assert current_y == 120
        assert current_x == 0

    async def test_advance_to_next_line_draws_ellipsis_when_exceeding_max_height(self):
        draw_text = DrawText(style=self.style)
        canvas = skia.Surface(1080, 1920).getCanvas()
        current_y, current_x = draw_text._advance_to_next_line(
            current_y=180,
            line_spacing=30,
            max_height=200,
            initial_x=0,
            canvas=canvas,
            font=skia.Font(),
            paint=skia.Paint(),
            current_x=50,
        )
        assert current_y == 200
        assert current_x == 0

    async def test_emoji_info_with_single_emoji(self):
        draw_text = DrawText(style=self.style)
        offset, character, font = draw_text._handle_emoji(0, {0: [2, "😊"]})
        assert offset == 2
        assert character == "😊"
        assert font == draw_text.emoji_font

    async def test_emoji_info_with_multiple_emojis(self):
        draw_text = DrawText(style=self.style)
        offset, character, font = draw_text._handle_emoji(0, {0: [2, "😊"], 2: [4, "🌍"]})
        assert offset == 2
        assert character == "😊"
        assert font == draw_text.emoji_font

    async def test_emoji_info_with_no_emoji(self):
        draw_text = DrawText(style=self.style)
        with pytest.raises(ParseError):
            _ = draw_text._handle_emoji(0, {})


@pytest.mark.asyncio
class TestPasteFunction:
    async def async_setup(self):
        self.canvas: MagicMock = MagicMock(spec=skia.Canvas)
        self.target: MagicMock = MagicMock(spec=skia.Image)
        self.position = (10, 20)
        self.target.dimensions.return_value.fWidth = 100
        self.target.dimensions.return_value.fHeight = 200

    async def test_paste_without_clear_background(self):
        await self.async_setup()
        await paste(self.canvas, self.target, self.position, clear_background=False)

        img_width = self.target.dimensions().fWidth
        img_height = self.target.dimensions().fHeight
        rec = skia.Rect.MakeXYWH(*self.position, img_width, img_height)

        self.canvas.drawImageRect.assert_called_once_with(self.target, skia.Rect(0, 0, img_width, img_height), rec)

    async def test_paste_with_clear_background(self):
        await self.async_setup()
        await paste(self.canvas, self.target, self.position, clear_background=True)

        img_width = self.target.dimensions().fWidth
        img_height = self.target.dimensions().fHeight
        rec = skia.Rect.MakeXYWH(*self.position, img_width, img_height)

        self.canvas.save.assert_called_once()
        self.canvas.clipRect.assert_called_once_with(rec, skia.ClipOp.kIntersect)
        self.canvas.clear.assert_called_once_with(skia.Color(255, 255, 255, 0))
        self.canvas.drawImageRect.assert_called_once_with(self.target, skia.Rect(0, 0, img_width, img_height), rec)
        self.canvas.restore.assert_called_once()

    async def test_paste_logs_attribute_error(self, caplog):
        await self.async_setup()
        canvas = None
        with caplog.at_level("ERROR"):
            await paste(canvas, self.target, self.position)  # type: ignore
        assert any("Failed to paste image" in record.message for record in caplog.records)


@pytest.mark.asyncio
class TestDrawTextFunction:
    async def async_setup(self, text="Hello, world!", emoji_info=None):
        self.canvas = MagicMock(spec=skia.Canvas)
        self.text = text
        self.font_size = 20
        self.position_and_bounds = (10, 20, 200, 100, 5)
        self.font_color = (0, 0, 0, 255)

        font_family = "Noto Sans SC"
        emoji_font_family = "Noto Color Emoji"
        font_style = "Normal"
        self.draw_text_instance = DrawText(SetDynStyle(font_family, emoji_font_family, font_style).set_style)

        # 模拟方法
        self.draw_text_instance.set_font_sizes = MagicMock()
        self.draw_text_instance.initialize_paint = MagicMock(
            return_value=skia.Paint(AntiAlias=True, Color=skia.Color(*self.font_color))
        )
        self.draw_text_instance.extract_emoji_info = AsyncMock(return_value=(self.text, emoji_info or {}))
        self.draw_text_instance._handle_emoji = MagicMock(return_value=(7, "🌍", skia.Font()))
        self.draw_text_instance._font_contains_character = MagicMock(return_value=True)

        # 使用实际的 skia.Font 对象
        self.actual_font = skia.Font()
        self.draw_text_instance.match_font = MagicMock(return_value=self.actual_font)
        self.draw_text_instance.text_font = self.actual_font

        # 包装 measureText 行为
        self.original_measure_text = self.actual_font.measureText

        def measure_text_wrapper(_text):
            return 50 if "🌍" in _text else 500

        self.measure_text_mock = patch.object(skia.Font, "measureText", side_effect=measure_text_wrapper)
        self.measure_text_mock.start()

        self.draw_text_instance._needs_new_line = MagicMock(side_effect=lambda x, _: x > 200)
        self.draw_text_instance._advance_to_next_line = MagicMock(return_value=(80, 10))

    async def _teardown_method(self):
        self.measure_text_mock.stop()

    async def test_draw_text(self):
        await self.async_setup()
        await self.draw_text_instance.draw_text(
            self.canvas, self.text, self.font_size, self.position_and_bounds, self.font_color
        )

        self.draw_text_instance.set_font_sizes.assert_called_once_with(self.font_size)  # type: ignore
        self.draw_text_instance.initialize_paint.assert_called_once_with(self.font_color)  # type: ignore
        self.draw_text_instance.extract_emoji_info.assert_called_once_with(self.text)  # type: ignore
        assert self.draw_text_instance._handle_emoji.call_count == 0  # type: ignore
        assert self.draw_text_instance._font_contains_character.call_count > 0  # type: ignore
        assert self.draw_text_instance._needs_new_line.call_count > 0  # type: ignore

        await self._teardown_method()

    async def test_draw_text_with_newline(self):
        await self.async_setup(text="Hello,\nworld!")
        await self.draw_text_instance.draw_text(
            self.canvas, self.text, self.font_size, self.position_and_bounds, self.font_color
        )

        assert self.canvas.drawTextBlob.call_count == 6

        await self._teardown_method()

    async def test_draw_text_with_emoji(self):
        await self.async_setup(text="Hello 🌍", emoji_info={6: [7, "🌍"]})
        await self.draw_text_instance.draw_text(
            self.canvas, self.text, self.font_size, self.position_and_bounds, self.font_color
        )

        assert self.draw_text_instance._handle_emoji.call_count == 1  # type: ignore

        await self._teardown_method()

    async def test_draw_text_with_wrapping(self):
        await self.async_setup()
        self.draw_text_instance._needs_new_line = MagicMock(side_effect=lambda x, _: x > 50)  # 小于50就换行

        await self.draw_text_instance.draw_text(
            self.canvas, self.text, self.font_size, self.position_and_bounds, self.font_color
        )

        assert self.draw_text_instance._advance_to_next_line.call_count > 0  # type: ignore

        await self._teardown_method()

    async def test_draw_text_exceeds_max_height(self):
        await self.async_setup()
        self.draw_text_instance._advance_to_next_line = MagicMock(side_effect=lambda *args: (args[0] + 100, args[1]))

        await self.draw_text_instance.draw_text(
            self.canvas, self.text, self.font_size, self.position_and_bounds, self.font_color
        )

        assert self.draw_text_instance._advance_to_next_line.call_count > 0
        assert self.canvas.drawTextBlob.call_count < len(self.text)

        await self._teardown_method()


class TestMatchFontMethod:
    @pytest.fixture(autouse=True)
    def _setup_method(self):
        font_family = "Noto Sans SC"
        emoji_font_family = "Noto Color Emoji"
        font_style = "Normal"
        self.draw_text_instance = DrawText(SetDynStyle(font_family, emoji_font_family, font_style).set_style)

    def test_match_font_with_existing_character(self):
        char = "A"
        font_size = 20

        typeface = skia.Typeface.MakeDefault()  # type: ignore

        with patch.object(skia.FontMgr, "matchFamilyStyleCharacter", return_value=typeface):
            matched_font = self.draw_text_instance.match_font(char, font_size)
            assert matched_font is not None
            assert matched_font.getSize() == font_size

    def test_match_font_with_non_existing_character(self):
        char = "\u4e00"
        font_size = 20

        with patch.object(skia.FontMgr, "matchFamilyStyleCharacter", return_value=None):
            matched_font = self.draw_text_instance.match_font(char, font_size)
            assert matched_font is None


class TestSetFontSizesMethod:
    @pytest.fixture(autouse=True)
    def _setup_method(self):
        font_family = "Noto Sans SC"
        emoji_font_family = "Noto Color Emoji"
        font_style = "Normal"
        self.draw_text_instance = DrawText(SetDynStyle(font_family, emoji_font_family, font_style).set_style)

        self.text_font_mock = MagicMock(spec=skia.Font)
        self.emoji_font_mock = MagicMock(spec=skia.Font)
        self.draw_text_instance.text_font = self.text_font_mock
        self.draw_text_instance.emoji_font = self.emoji_font_mock

    def test_set_font_sizes(self):
        font_size = 24
        self.draw_text_instance.set_font_sizes(font_size)

        self.text_font_mock.setSize.assert_called_once_with(font_size)
        self.emoji_font_mock.setSize.assert_called_once_with(font_size)
