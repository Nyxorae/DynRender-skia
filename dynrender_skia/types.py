"""Shared type definitions."""

from typing import Protocol, Optional
import numpy as np
import skia


class Renderable(Protocol):
    """Protocol for objects that can be rendered to a numpy array."""

    async def run(self, repost: bool = False) -> Optional[np.ndarray]:
        ...


class Renderer(Protocol):
    """Protocol for renderer classes."""

    async def run(self, data, repost: bool = False) -> Optional[np.ndarray]:
        ...
