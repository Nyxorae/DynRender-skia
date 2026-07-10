"""Strategy Registry — decorator-based renderer dispatch.

**Design pattern: Strategy + Registry**

Instead of a long ``if/elif`` chain::

    if major_type == "MAJOR_TYPE_DRAW":
        return DynMajorDraw(...)
    elif major_type == "MAJOR_TYPE_ARCHIVE":
        return DynMajorArchive(...)
    # ... 14 more branches ...

Each renderer *self-registers* with a decorator::

    @register_major("MAJOR_TYPE_DRAW")
    class MajorDraw(BaseMajorRenderer):
        ...

The registry is populated when the module is imported (triggered by
``renderers/__init__.py``).  Lookup is O(1) via a plain ``dict``.
"""

from typing import Optional

# ---- Module-level registries ------------------------------------------

_major_renderers: dict[str, type] = {}       # e.g. "MAJOR_TYPE_DRAW" → MajorDraw
_additional_renderers: dict[str, type] = {}   # e.g. "ADDITIONAL_TYPE_VOTE" → AdditionalVote


# ---- Decorators -------------------------------------------------------

def register_major(key: str):
    """Class decorator — register a major-type renderer under *key*."""
    def decorator(cls):
        _major_renderers[key] = cls
        return cls
    return decorator


def register_additional(key: str):
    """Class decorator — register an additional-card renderer under *key*."""
    def decorator(cls):
        _additional_renderers[key] = cls
        return cls
    return decorator


# ---- Lookup -----------------------------------------------------------

def get_major_renderer(key: str) -> Optional[type]:
    """Return the renderer class for *key*, or ``None`` if unsupported."""
    return _major_renderers.get(key)


def get_additional_renderer(key: str) -> Optional[type]:
    """Return the renderer class for *key*, or ``None`` if unsupported."""
    return _additional_renderers.get(key)
