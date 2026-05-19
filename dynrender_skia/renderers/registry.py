"""Strategy registry for renderer type dispatch.

Replaces the manual if/elif chains in BiliMajor and BiliAdditional
with decorator-based registration.
"""

from typing import Optional

_major_renderers: dict[str, type] = {}
_additional_renderers: dict[str, type] = {}


def register_major(key: str):
    def decorator(cls):
        _major_renderers[key] = cls
        return cls
    return decorator


def register_additional(key: str):
    def decorator(cls):
        _additional_renderers[key] = cls
        return cls
    return decorator


def get_major_renderer(key: str) -> Optional[type]:
    return _major_renderers.get(key)


def get_additional_renderer(key: str) -> Optional[type]:
    return _additional_renderers.get(key)
