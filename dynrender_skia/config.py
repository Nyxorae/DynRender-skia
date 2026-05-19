"""Style configuration and static file initialization."""

import json
from os import getcwd, makedirs, path
from typing import Any, Optional
from zipfile import ZipFile

from pydantic import BaseModel

try:
    import skia
except ImportError as e:
    import sys
    print(
        "Missing dependent files. Please install dependence:\n"
        "  Ubuntu: apt install libgl1-mesa-glx\n"
        "  ArchLinux: pacman -S libgl\n"
        "  Centos: yum install mesa-libGL -y"
    )
    sys.exit(1)


# ---- Style models ----

class FontSize(BaseModel):
    text: int
    name: int
    time: int
    title: int
    sub_title: int


class FontColor(BaseModel):
    text: tuple
    title: tuple
    name_big_vip: tuple
    name_small_vip: tuple
    rich_text: tuple
    sub_title: tuple
    white: tuple


class BackgroundColor(BaseModel):
    normal: tuple
    repost: tuple
    border: tuple


class FontCfg(BaseModel):
    font_family: str
    emoji_font_family: str
    font_style: Any
    font_size: FontSize


class ColorCfg(BaseModel):
    font_color: FontColor
    background: BackgroundColor


class PolyStyle(BaseModel):
    font: FontCfg
    color: ColorCfg


# ---- Style factory ----

_FONT_STYLE_MAP = {
    "Normal": skia.FontStyle().Normal(),
    "Bold": skia.FontStyle().Bold(),
    "Italic": skia.FontStyle().Italic(),
    "BoldItalic": skia.FontStyle().BoldItalic(),
}

_DEFAULT_STYLE_CFG = {
    "color": {
        "font_color": {
            "text": (0, 0, 0, 255),
            "sub_title": (153, 162, 170, 255),
            "title": (0, 0, 0, 255),
            "name_big_vip": (251, 107, 148, 255),
            "name_small_vip": (60, 232, 78, 255),
            "rich_text": (0, 161, 214, 255),
            "white": (255, 255, 255, 255),
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
    cfg = _DEFAULT_STYLE_CFG.copy()
    cfg["font"] = {
        **cfg["font"],
        "font_family": font_family,
        "emoji_font_family": emoji_font_family,
        "font_style": _FONT_STYLE_MAP.get(font_style, skia.FontStyle().Normal()),
    }
    return PolyStyle(**cfg)


# ---- Static file initialization ----

def init_static_path(data_path: Optional[str] = None) -> str:
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
    file = ZipFile(path.join(src_dir, "Static.zip"))
    file.extractall(target_dir)


def _sync_font_cache(static_path: str) -> None:
    font_cache_path = path.join(static_path, "font_family.json")
    new_font_list = list(skia.FontMgr())
    if not path.exists(font_cache_path):
        with open(font_cache_path, "w") as f:
            f.write(json.dumps(new_font_list, ensure_ascii=False))
    else:
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
