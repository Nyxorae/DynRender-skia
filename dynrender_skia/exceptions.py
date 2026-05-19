"""Exception hierarchy for dynrender-skia."""

from typing import Optional


class DynRenderException(Exception):
    """Base exception for this module."""

    def __str__(self) -> str:
        return self.__repr__()


class SkiaBaseError(DynRenderException):
    """Skia related error."""

    def __init__(self, message: Optional[str] = None, status: int = 0) -> None:
        self.message = message
        self.status = status

    def __repr__(self) -> str:
        return f"SkiaBaseError(status={self.status}, message={self.message})"


class ImageDecodeError(SkiaBaseError):
    """Image decoding failed."""


class DrawingError(SkiaBaseError):
    """Drawing operation failed."""


class ParseError(SkiaBaseError):
    """Parsing operation failed."""
