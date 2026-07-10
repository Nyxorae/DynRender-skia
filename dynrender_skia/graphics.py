"""Common graphics primitives — re-export hub.

Image fetching, compositing, shape drawing, and text layout primitives
are organised into focused sub-modules for maintainability.
"""

from ._io import fetch_images, request_img  # noqa: F401 — re-export
from .composite import merge_pictures, paste  # noqa: F401 — re-export
from .shapes import circle_crop, draw_shadow, make_badge, round_corners  # noqa: F401 — re-export
from .text_drawer import TextDrawer  # noqa: F401 — re-export for backward compat
