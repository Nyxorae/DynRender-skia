"""Renderer modules."""

from .registry import get_major_renderer, get_additional_renderer
from .header import BiliHeader, RepostHeader
from .footer import Footer
from .text import BiliText
from .repost import BiliRepost

# Trigger decorator registration for all major and additional types
from . import major  # noqa: F401
from . import additional  # noqa: F401

__all__ = [
    "BiliHeader",
    "RepostHeader",
    "Footer",
    "BiliText",
    "BiliRepost",
    "get_major_renderer",
    "get_additional_renderer",
]
