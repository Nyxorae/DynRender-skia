"""Renderer registration and instantiation tests."""

import pytest

from dynrender_skia.config import create_style
from dynrender_skia.renderers.registry import (
    _additional_renderers,
    _major_renderers,
)


KNOWN_MAJOR_TYPES = [
    "MAJOR_TYPE_DRAW",
    "MAJOR_TYPE_ARCHIVE",
    "MAJOR_TYPE_LIVE_RCMD",
    "MAJOR_TYPE_ARTICLE",
    "MAJOR_TYPE_COMMON",
    "MAJOR_TYPE_MUSIC",
    "MAJOR_TYPE_PGC",
    "MAJOR_TYPE_MEDIALIST",
    "MAJOR_TYPE_COURSES",
    "MAJOR_TYPE_UGC_SEASON",
    "MAJOR_TYPE_LIVE",
    "MAJOR_TYPE_OPUS",
    "MAJOR_TYPE_NONE",
    "MAJOR_TYPE_BLOCKED",
]

KNOWN_ADDITIONAL_TYPES = [
    "ADDITIONAL_TYPE_RESERVE",
    "ADDITIONAL_TYPE_UPOWER_LOTTERY",
    "ADDITIONAL_TYPE_GOODS",
    "ADDITIONAL_TYPE_UGC",
    "ADDITIONAL_TYPE_VOTE",
    "ADDITIONAL_TYPE_COMMON",
]


@pytest.fixture(scope="module")
def style():
    return create_style()


class TestRendererRegistry:
    def test_all_major_types_registered(self) -> None:
        for key in KNOWN_MAJOR_TYPES:
            assert key in _major_renderers, f"Missing major renderer: {key}"

    def test_all_additional_types_registered(self) -> None:
        for key in KNOWN_ADDITIONAL_TYPES:
            assert key in _additional_renderers, f"Missing additional renderer: {key}"

    def test_no_unknown_types_leaked(self) -> None:
        assert set(_major_renderers.keys()) == set(KNOWN_MAJOR_TYPES)
        assert set(_additional_renderers.keys()) == set(KNOWN_ADDITIONAL_TYPES)


class TestMajorRendererInstantiation:
    def test_major_renderers_can_be_instantiated(self, style) -> None:
        from dynrender_skia.renderers.registry import get_major_renderer

        for key in KNOWN_MAJOR_TYPES:
            cls = get_major_renderer(key)
            assert cls is not None, f"No renderer for {key}"
            _ = cls("stub_src_path", style)

    def test_additional_renderers_can_be_instantiated(self, style) -> None:
        from dynrender_skia.renderers.registry import get_additional_renderer

        for key in KNOWN_ADDITIONAL_TYPES:
            cls = get_additional_renderer(key)
            assert cls is not None, f"No renderer for {key}"
            cls("stub_src_path", style, None)


class TestCoreRendererInstantiation:
    def test_header_can_be_instantiated(self, style) -> None:
        from dynrender_skia.renderers.header import BiliHeader, RepostHeader
        BiliHeader("stub", style)
        RepostHeader("stub", style)

    def test_text_can_be_instantiated(self, style) -> None:
        from dynrender_skia.renderers.text import BiliText
        BiliText("stub", style)

    def test_footer_can_be_instantiated(self, style) -> None:
        from dynrender_skia.renderers.footer import Footer
        Footer("stub", style)

    def test_repost_can_be_instantiated(self, style) -> None:
        from dynrender_skia.renderers.repost import BiliRepost
        BiliRepost("stub", style)
