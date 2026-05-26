"""Renderer modules."""

# Trigger decorator registration for all major and additional types
from . import (
    additional,  # noqa: F401
    major,  # noqa: F401
)
from .base import safe_run
from .footer import Footer
from .header import BiliHeader, RepostHeader
from .registry import get_additional_renderer, get_major_renderer
from .repost import BiliRepost
from .text import BiliText

__all__ = [
    "BiliHeader",
    "BiliRepost",
    "BiliText",
    "Footer",
    "RepostHeader",
    "get_additional_renderer",
    "get_major_renderer",
    "safe_run",
]
