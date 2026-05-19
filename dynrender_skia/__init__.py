"""DynRender-skia: Bilibili dynamic content renderer using Skia."""


def __getattr__(name):
    if name == "DynRender":
        from .core import DynRender
        return DynRender
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["DynRender"]
