"""Style configuration and static-file bootstrap.

Design: **Builder** — ``create_style()`` assembles a ``PolyStyle``
from defaults overlaid with user preferences.  The static-file
initialisation auto-extracts the bundled ``Static.zip`` on first run.
"""

import json
from os import getcwd, makedirs, path
from typing import Any, Optional
from zipfile import ZipFile

from pydantic import BaseModel

try:
    import skia
except ImportError:
    import sys
    print(
        "Missing Skia native library. Please install:\n"
        "  Ubuntu:   apt install libgl1-mesa-glx\n"
        "  Arch:     pacman -S libgl\n"
        "  CentOS:   yum install mesa-libGL -y"
    )
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════
# Style data models (Pydantic — validated, self-documenting config)
# ═══════════════════════════════════════════════════════════════════════


class FontSize(BaseModel):
    """Font size presets (px)."""
    text: int       # body text (40)
    name: int       # username (45)
    time: int       # timestamp (35)
    title: int      # secondary title (30)
    sub_title: int  # tertiary label (20)


class FontColor(BaseModel):
    """RGBA colour palette for text."""
    text: tuple             # primary body text
    title: tuple            # titles
    name_big_vip: tuple     # pink — annual VIP
    name_small_vip: tuple   # green — monthly VIP
    rich_text: tuple        # blue — @mentions, #topics, links
    sub_title: tuple        # grey — secondary labels
    white: tuple            # white on dark backgrounds


class BackgroundColor(BaseModel):
    """Background colour per rendering context."""
    normal: tuple  # white (standalone post)
    repost: tuple  # light grey (nested forward)
    border: tuple  # card border


class FontCfg(BaseModel):
    """Font family and style configuration."""
    font_family: str
    emoji_font_family: str
    font_style: Any          # skia.FontStyle
    font_size: FontSize


class ColorCfg(BaseModel):
    """Complete colour configuration."""
    font_color: FontColor
    background: BackgroundColor


class PolyStyle(BaseModel):
    """Root style object — passed to every renderer."""
    font: FontCfg
    color: ColorCfg


# ═══════════════════════════════════════════════════════════════════════
# Style factory (Builder pattern)
# ═══════════════════════════════════════════════════════════════════════

_FONT_STYLE_MAP = {
    "Normal":     skia.FontStyle().Normal(),
    "Bold":       skia.FontStyle().Bold(),
    "Italic":     skia.FontStyle().Italic(),
    "BoldItalic": skia.FontStyle().BoldItalic(),
}

# Bilibili-branded colour palette and typography defaults
_DEFAULT_STYLE_CFG = {
    "color": {
        "font_color": {
            "text":             (0, 0, 0, 255),
            "sub_title":        (153, 162, 170, 255),
            "title":            (0, 0, 0, 255),
            "name_big_vip":     (251, 107, 148, 255),   # B站大会员粉
            "name_small_vip":   (60, 232, 78, 255),     # B站小会员绿
            "rich_text":        (0, 161, 214, 255),     # B站链接蓝
            "white":            (255, 255, 255, 255),
        },
        "background": {
            "normal": (255, 255, 255, 255),
            "repost": (241, 242, 243, 255),
            "border": (229, 233, 239, 255),
        },
    },
    "font": {
        "font_size": {
            "name": 45,
            "text": 40,
            "time": 35,
            "title": 30,
            "sub_title": 20,
        },
    },
}


def create_style(
    font_family: str = "Noto Sans SC",
    emoji_font_family: str = "Noto Color Emoji",
    font_style: str = "Normal",
) -> PolyStyle:
    """Build a ``PolyStyle`` with the given font preferences.

    Defaults are merged with user-provided values; unspecified fields
    retain the Bilibili-branded palette above.
    """
    cfg = _DEFAULT_STYLE_CFG.copy()
    cfg["font"] = {
        **cfg["font"],
        "font_family": font_family,
        "emoji_font_family": emoji_font_family,
        "font_style": _FONT_STYLE_MAP.get(font_style, skia.FontStyle().Normal()),
    }
    return PolyStyle(**cfg)


# ═══════════════════════════════════════════════════════════════════════
# Static file bootstrap
# ═══════════════════════════════════════════════════════════════════════


def init_static_path(data_path: Optional[str] = None) -> str:
    """Ensure static assets exist and return their directory path.

    If *data_path* is given, assets are extracted there.  Otherwise
    they are extracted into ``cwd/Static``.  A font cache file is
    written so callers can discover available system fonts.
    """
    current_dir = path.dirname(path.abspath(__file__))

    if data_path is None:
        program_running_path = getcwd()
        static_path = path.join(program_running_path, "Static")
        if not path.exists(static_path):
            _unzip_static(current_dir, program_running_path)
    else:
        static_path = path.join(data_path, "Static")
        if not path.exists(data_path):
            makedirs(data_path)
        if not path.exists(static_path):
            _unzip_static(current_dir, data_path)

    _sync_font_cache(static_path)
    return static_path


def _unzip_static(src_dir: str, target_dir: str) -> None:
    """Extract the bundled ``Static.zip`` to *target_dir*."""
    with ZipFile(path.join(src_dir, "Static.zip")) as zf:
        zf.extractall(target_dir)


def _sync_font_cache(static_path: str) -> None:
    """Write/update ``font_family.json`` enumerating installed system fonts."""
    font_cache_path = path.join(static_path, "font_family.json")
    new_font_list = list(skia.FontMgr())

    if not path.exists(font_cache_path):
        with open(font_cache_path, "w") as f:
            f.write(json.dumps(new_font_list, ensure_ascii=False))
        return

    with open(font_cache_path, "r+") as f:
        old_data = f.read()
        if old_data:
            old_font_list = json.loads(old_data)
            if new_font_list != old_font_list:
                f.seek(0)
                f.truncate()
                f.write(json.dumps(new_font_list, ensure_ascii=False))
        else:
            f.write(json.dumps(new_font_list, ensure_ascii=False))
