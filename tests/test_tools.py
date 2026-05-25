import pathlib
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import httpx
import numpy as np
import pytest
import pytest_asyncio
import respx
import skia

from dynrender_skia.config import create_style
from dynrender_skia.graphics import merge_pictures, request_img, fetch_images, TextDrawer, paste
from dynrender_skia.exceptions import ParseError
from dynrender_skia.typesetter import Atom, CharClass


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
    with respx.mock() as mock:
        url = "http://bilibili.com"

        mock.get(url).mock(side_effect=httpx.ConnectError("Connection error"))

        async with httpx.AsyncClient() as client:
            img = await request_img(client, url, None)
            assert img is None


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

    async def test_request_img_with_http_exception(self, client: httpx.AsyncClient, mock_img_url: str, caplog) -> None:
        async with respx.mock(base_url=mock_img_url) as mock:
            mock.get(mock_img_url).mock(side_effect=httpx.ConnectError("Connection error"))

            _ = await request_img(client, mock_img_url, None)
            assert "Request or HTTP error occurred" in caplog.text

    async def test_skia_encode_exception(self, client: httpx.AsyncClient, mock_img_url: str, caplog) -> None:
        async with respx.mock() as mock:
            img_content = None
            mock.get(mock_img_url).respond(content=img_content, status_code=200)
            _ = await request_img(client, mock_img_url, None)
            assert "Image decode error or request returned none in content" in caplog.text

    async def test_unexpected_exception(self, client: httpx.AsyncClient, mock_img_url: str, caplog):
        async with respx.mock(base_url=mock_img_url) as mock:
            mock.get(mock_img_url).mock(side_effect=TypeError("Invalid type provided"))

            _ = await request_img(client, mock_img_url, None)
            assert "Unexpected error" in caplog.text


@pytest.mark.asyncio
class TestGetPictures:
    @pytest.fixture
    def mock_skia_image(self, img_path: Path):
        return skia.Image.MakeFromEncoded(encoded=img_path.read_bytes())  # type: ignore

    async def test_fetch_images_with_single_url(
        self, mock_img_url: str, img_path: pathlib.Path, mock_skia_image: skia.Image
    ) -> None:
        async with respx.mock(base_url=mock_img_url) as mock:
            img_content = img_path.read_bytes()
            mock.get(mock_img_url).respond(content=img_content, status_code=200)
            image: skia.Image = await fetch_images(mock_img_url, None)

            img_array = image.tobytes()
            result = mock_skia_image.tobytes()
            assert img_array == result

    async def test_fetch_images_with_multiple_urls(self, mock_img_url: str, img_path: pathlib.Path) -> None:
        async with respx.mock(base_url=mock_img_url) as mock:
            img_content = img_path.read_bytes()
            mock.get(mock_img_url).respond(content=img_content, status_code=200)

            img_list = await fetch_images([mock_img_url, mock_img_url], None)
            assert len(img_list) == 2
            assert all(img is not None for img in img_list)

    async def test_initialize_paint(self):
        font_color = (255, 0, 0, 255)

        paint = TextDrawer.initialize_paint(font_color)

        assert isinstance(paint, skia.Paint)
        assert paint.isAntiAlias() is True, "Paint should enabled anti-alias"
        assert paint.getColor() == skia.Color(*font_color), "Paint color should be the same as font color"

    async def test_draw_ellipsis(self, mocker):
        x, y = 100, 50
        font = skia.Font(skia.Typeface("Arial"), 20)
        font_color = (0, 0, 0, 255)
        paint = TextDrawer.initialize_paint(font_color)

        canvas = mocker.MagicMock(spec=skia.Canvas)
        mock_text_blob = mocker.patch("skia.TextBlob")

        TextDrawer.draw_ellipsis(canvas, x, y, font, paint)

        mock_text_blob.assert_called_once_with("...", font)
        canvas.drawTextBlob.assert_called_once_with(mock_text_blob.return_value, x, y, paint)


@pytest.mark.asyncio
class TestTextDrawer:
    @pytest_asyncio.fixture(autouse=True)
    async def setup(self):
        font_family = "Noto Sans SC"
        emoji_font_family = "Noto Color Emoji"
        font_style = "Normal"
        self.style = create_style(font_family, emoji_font_family, font_style)

    async def test_extract_emoji_info_removes_tabs_and_extracts_emoji(self):
        draw_text = TextDrawer(style=self.style)
        text = "Hello,\t🌍!"
        cleaned_text, emoji_info = await draw_text.extract_emoji_info(text)
        assert cleaned_text == "Hello,🌍!"
        assert emoji_info == {6: [7, "🌍"]}

    async def test_extract_emoji_info_handles_no_emoji(self):
        draw_text = TextDrawer(style=self.style)
        text = "Hello, World!"
        cleaned_text, emoji_info = await draw_text.extract_emoji_info(text)
        assert cleaned_text == "Hello, World!"
        assert emoji_info == {}

    async def test_extract_emoji_info_handles_multiple_emojis(self):
        draw_text = TextDrawer(style=self.style)
        text = "Hello, 🌍! How are you? 😊"
        cleaned_text, emoji_info = await draw_text.extract_emoji_info(text)
        assert cleaned_text == "Hello, 🌍! How are you? 😊"
        assert emoji_info == {7: [8, "🌍"], 23: [24, "😊"]}

    async def test_extract_emoji_info_handles_only_tabs(self):
        draw_text = TextDrawer(style=self.style)
        text = "\t\t\t"
        cleaned_text, emoji_info = await draw_text.extract_emoji_info(text)
        assert cleaned_text == ""
        assert emoji_info == {}

    async def test_needs_new_line_returns_true_when_exceeding_max_width(self):
        assert TextDrawer._needs_new_line(1100, 1080) is True

    async def test_needs_new_line_returns_false_when_within_max_width(self):
        assert TextDrawer._needs_new_line(900, 1080) is False

    async def test_needs_new_line_returns_true_when_equal_to_max_width(self):
        assert TextDrawer._needs_new_line(1080, 1080) is False

    async def test_font_contains_character_returns_true_for_valid_character(self):
        font = skia.Font(skia.Typeface.MakeDefault(), 12)  # type: ignore
        assert TextDrawer._font_contains_character(font, "A") is True

    async def test_font_contains_character_returns_false_for_invalid_character(self):
        font = skia.Font(skia.Typeface.MakeDefault(), 12)  # type: ignore
        assert TextDrawer._font_contains_character(font, "\uffff") is False

    async def test_advance_to_next_line_adds_line_spacing_when_within_max_height(self):
        draw_text = TextDrawer(style=self.style)
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
        draw_text = TextDrawer(style=self.style)
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
        draw_text = TextDrawer(style=self.style)
        offset, character, font = draw_text._handle_emoji(0, {0: [2, "😊"]})
        assert offset == 2
        assert character == "😊"
        assert font == draw_text.emoji_font

    async def test_emoji_info_with_multiple_emojis(self):
        draw_text = TextDrawer(style=self.style)
        offset, character, font = draw_text._handle_emoji(0, {0: [2, "😊"], 2: [4, "🌍"]})
        assert offset == 2
        assert character == "😊"
        assert font == draw_text.emoji_font

    async def test_emoji_info_with_no_emoji(self):
        draw_text = TextDrawer(style=self.style)
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
class TestTextDrawerFunction:
    async def async_setup(self, text="Hello, world!", emoji_info=None):
        self.canvas = MagicMock(spec=skia.Canvas)
        self.text = text
        self.font_size = 20
        self.position_and_bounds = (10, 20, 200, 100, 5)
        self.font_color = (0, 0, 0, 255)

        font_family = "Noto Sans SC"
        emoji_font_family = "Noto Color Emoji"
        font_style = "Normal"
        self.draw_text_instance = TextDrawer(create_style(font_family, emoji_font_family, font_style))

        # Mock methods that draw_text calls directly
        self.draw_text_instance.set_font_sizes = MagicMock()
        self.mock_paint = skia.Paint(AntiAlias=True, Color=skia.Color(*self.font_color))
        self.draw_text_instance.initialize_paint = MagicMock(return_value=self.mock_paint)
        self.draw_text_instance.get_emoji_text = AsyncMock(return_value=emoji_info or {})
        self.draw_text_instance._font_contains_character = MagicMock(return_value=True)
        self.draw_text_instance.match_font = MagicMock(return_value=skia.Font())
        self.draw_text_instance.draw_ellipsis = MagicMock()

    @staticmethod
    def _make_atoms(text: str, *, width: float = 10.0) -> list[Atom]:
        """Build Atom objects simulating what atomize_text produces for plain text."""
        atoms: list[Atom] = []
        for ch in text:
            if ch == "\n":
                atoms.append(Atom(text="\n", width=0.0, char_class=CharClass.MANDATORY_BREAK))
            else:
                atoms.append(Atom(text=ch, width=width, char_class=CharClass.ALPHABETIC))
        return atoms

    @staticmethod
    def _mock_typesetter(atoms: list[Atom], lines: list[tuple[int, int, float]]):
        """Set up mocks for atomize_text and KnuthPlassLineBreaker. Returns stop-callables."""
        p_atomize = patch("dynrender_skia.typesetter.atomize_text", return_value=atoms)
        p_breaker = patch("dynrender_skia.typesetter.KnuthPlassLineBreaker")
        p_atomize.start()
        mock_breaker_class = p_breaker.start()
        mock_breaker = MagicMock()
        mock_breaker.break_lines.return_value = lines
        mock_breaker_class.return_value = mock_breaker
        return p_atomize, p_breaker

    async def test_draw_text(self):
        await self.async_setup()
        atoms = self._make_atoms(self.text)
        p1, p2 = self._mock_typesetter(atoms, [(0, len(atoms), 0.0)])

        try:
            await self.draw_text_instance.draw_text(
                self.canvas, self.text, self.font_size, self.position_and_bounds, self.font_color
            )

            self.draw_text_instance.set_font_sizes.assert_called_once_with(self.font_size)
            self.draw_text_instance.initialize_paint.assert_called_once_with(self.font_color)
            self.draw_text_instance.get_emoji_text.assert_called_once_with(self.text)
            assert self.canvas.drawTextBlob.call_count == len(atoms)
            assert self.draw_text_instance._font_contains_character.call_count > 0
        finally:
            p1.stop()
            p2.stop()

    async def test_draw_text_with_newline(self):
        await self.async_setup(text="Hello,\nworld!")
        atoms = self._make_atoms(self.text)
        # Two lines: first ends before \n, second starts after \n
        lines = [(0, 6, 0.0), (7, len(atoms), 0.0)]
        p1, p2 = self._mock_typesetter(atoms, lines)

        try:
            await self.draw_text_instance.draw_text(
                self.canvas, self.text, self.font_size, self.position_and_bounds, self.font_color
            )

            # 13 atoms total, but 1 is MANDATORY_BREAK (skipped) → 12 drawTextBlob calls
            assert self.canvas.drawTextBlob.call_count == len(atoms) - 1
        finally:
            p1.stop()
            p2.stop()

    async def test_draw_text_with_emoji(self):
        await self.async_setup(text="Hello 🌍", emoji_info={6: [7, "🌍"]})
        # Build atoms including an emoji atom
        atoms = [
            Atom(text="H", width=10.0, char_class=CharClass.ALPHABETIC),
            Atom(text="e", width=10.0, char_class=CharClass.ALPHABETIC),
            Atom(text="l", width=10.0, char_class=CharClass.ALPHABETIC),
            Atom(text="l", width=10.0, char_class=CharClass.ALPHABETIC),
            Atom(text="o", width=10.0, char_class=CharClass.ALPHABETIC),
            Atom(text=" ", width=10.0, char_class=CharClass.SPACE),
            Atom(text="🌍", width=50.0, char_class=CharClass.EMOJI),
        ]
        p1, p2 = self._mock_typesetter(atoms, [(0, len(atoms), 0.0)])

        try:
            await self.draw_text_instance.draw_text(
                self.canvas, self.text, self.font_size, self.position_and_bounds, self.font_color
            )

            assert self.draw_text_instance.get_emoji_text.call_count == 1
            assert self.canvas.drawTextBlob.call_count == len(atoms)
        finally:
            p1.stop()
            p2.stop()

    async def test_draw_text_with_wrapping(self):
        await self.async_setup()
        atoms = self._make_atoms(self.text)
        # Two lines to trigger wrapping
        lines = [(0, 5, 0.0), (5, len(atoms), 0.0)]
        p1, p2 = self._mock_typesetter(atoms, lines)

        try:
            await self.draw_text_instance.draw_text(
                self.canvas, self.text, self.font_size, self.position_and_bounds, self.font_color
            )

            # All atoms drawn across two lines
            assert self.canvas.drawTextBlob.call_count == len(atoms)
        finally:
            p1.stop()
            p2.stop()

    async def test_draw_text_exceeds_max_height(self):
        await self.async_setup()
        atoms = self._make_atoms(self.text)
        # y_bound=30, line_spacing=20: after first line current_y goes 20→40 >= 30 → ellipsis
        self.position_and_bounds = (10, 20, 200, 30, 20)
        lines = [(0, 4, 0.0), (4, len(atoms), 0.0)]
        p1, p2 = self._mock_typesetter(atoms, lines)

        try:
            await self.draw_text_instance.draw_text(
                self.canvas, self.text, self.font_size, self.position_and_bounds, self.font_color
            )

            # Only the first line's 4 atoms are drawn; second line is cut off by y_bound
            assert self.canvas.drawTextBlob.call_count == 4
            assert self.draw_text_instance.draw_ellipsis.call_count == 1
        finally:
            p1.stop()
            p2.stop()


class TestMatchFontMethod:
    @pytest.fixture(autouse=True)
    def _setup_method(self):
        font_family = "Noto Sans SC"
        emoji_font_family = "Noto Color Emoji"
        font_style = "Normal"
        self.draw_text_instance = TextDrawer(create_style(font_family, emoji_font_family, font_style))

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
        self.draw_text_instance = TextDrawer(create_style(font_family, emoji_font_family, font_style))

        self.text_font_mock = MagicMock(spec=skia.Font)
        self.emoji_font_mock = MagicMock(spec=skia.Font)
        self.draw_text_instance.text_font = self.text_font_mock
        self.draw_text_instance.emoji_font = self.emoji_font_mock

    def test_set_font_sizes(self):
        font_size = 24
        self.draw_text_instance.set_font_sizes(font_size)

        self.text_font_mock.setSize.assert_called_once_with(font_size)
        self.emoji_font_mock.setSize.assert_called_once_with(font_size)
