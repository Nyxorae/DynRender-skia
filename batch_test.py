"""Batch test: render various Bilibili dynamic types and generate a markdown report."""
import asyncio
import json
import sys
from datetime import datetime
from os import path

import httpx
import skia
from dynamicadaptor.DynamicConversion import formate_message
from dynrender_skia.Core import DynRender

COOKIE = ""  # Set your Bilibili cookie here if needed

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://t.bilibili.com/",
    "cookie": COOKIE,
}

DETAIL_URL = (
    "https://api.bilibili.com/x/polymer/web-dynamic/v1/detail"
    "?timezone_offset=-480&platform=web&gaia_source=main_web"
    "&id={}"
    "&features=itemOpusStyle,opusBigCover,onlyfansVote,endFooterHidden,"
    "decorationCard,onlyfansAssetsV2,ugcDelete,onlyfansQaCard,editable,"
    "opusPrivateVisible,avatarAutoTheme,sunflowerStyle,cardsEnhance,"
    "eva3CardOpus,eva3CardVideo,eva3CardComment,eva3CardVote,eva3CardUser"
    "&web_location=333.1368"
)
FEED_URL = "https://api.bilibili.com/x/polymer/web-dynamic/v1/feed/all?type=all&page={}"


async def fetch_dynamic(client: httpx.AsyncClient, did: str) -> dict | None:
    r = await client.get(DETAIL_URL.format(did), headers=HEADERS)
    data = r.json()
    if data.get("code") != 0:
        return None
    return data


async def fetch_feed(client: httpx.AsyncClient, page: int = 1) -> list[dict]:
    r = await client.get(FEED_URL.format(page), headers=HEADERS)
    data = r.json()
    return data.get("data", {}).get("items", [])


async def render_dynamic(engine: DynRender, item: dict) -> skia.Image | None:
    msg = await formate_message("web", item)
    if msg is None:
        return None
    arr = await engine.run(msg)
    if arr is None:
        return None
    return skia.Image.fromarray(arr, colorType=skia.ColorType.kRGBA_8888_ColorType)


async def main():
    engine = DynRender()
    out_dir = "test_output"
    import os
    os.makedirs(out_dir, exist_ok=True)

    async with httpx.AsyncClient(timeout=20) as client:
        # Step 1: Collect dynamic IDs from trending feed
        print("Fetching trending dynamics...")
        all_items = []
        for page in [1, 2, 3, 4, 5]:
            try:
                items = await fetch_feed(client, page)
                all_items.extend(items)
                print(f"  Page {page}: {len(items)} items")
            except Exception as e:
                print(f"  Page {page}: ERROR {e}")

        # Step 2: Identify unique types and pick samples
        type_samples: dict[str, tuple[str, str, dict]] = {}  # type -> (id, author, item)
        type_descriptions = {
            "DYNAMIC_TYPE_DRAW": "图文/画册",
            "DYNAMIC_TYPE_WORD": "纯文字",
            "DYNAMIC_TYPE_FORWARD": "转发",
            "DYNAMIC_TYPE_AV": "视频投稿",
            "DYNAMIC_TYPE_ARTICLE": "专栏文章",
            "DYNAMIC_TYPE_LIVE_RCMD": "直播推荐",
            "DYNAMIC_TYPE_PGC": "番剧/影视",
            "DYNAMIC_TYPE_MUSIC": "音乐",
            "DYNAMIC_TYPE_COMMON": "通用卡片",
            "DYNAMIC_TYPE_LIVE": "直播",
            "DYNAMIC_TYPE_MEDIALIST": "合集",
            "DYNAMIC_TYPE_COURSES": "课程",
            "DYNAMIC_TYPE_UGC_SEASON": "合集季",
        }

        for item in all_items:
            dtype = item.get("type", "")
            if dtype not in type_samples:
                author = item.get("modules", {}).get("module_author", {}).get("name", "unknown")
                did = item.get("id_str", "")
                if did:
                    # Re-fetch with full-featured URL to get opus-style data
                    data = await fetch_dynamic(client, did)
                    if data:
                        item = data["data"]["item"]
                    type_samples[dtype] = (did, author, item)

        # Also add known IDs for types we might have missed
        known_ids = [
            ("DYNAMIC_TYPE_DRAW", "1202655761342136353"),
            ("DYNAMIC_TYPE_DRAW", "1203956062672125953"),
            ("DYNAMIC_TYPE_FORWARD", "1202997378741698565"),
            ("DYNAMIC_TYPE_FORWARD", "1203768329657909271"),
            ("DYNAMIC_TYPE_AV", "1203954963152109653"),
            ("DYNAMIC_TYPE_LIVE_RCMD", "1203942447657254912"),
            # Additional known types to try
            ("DYNAMIC_TYPE_WORD", "1202895823712378887"),
            ("DYNAMIC_TYPE_ARTICLE", "1100497824657350657"),
        ]
        for dtype, did in known_ids:
            if dtype not in type_samples:
                try:
                    data = await fetch_dynamic(client, did)
                    if data:
                        item = data["data"]["item"]
                        actual_type = item.get("type", "")
                        author = item.get("modules", {}).get("module_author", {}).get("name", "unknown")
                        # Use actual type from API, not the expected one
                        if actual_type not in type_samples:
                            type_samples[actual_type] = (did, author, item)
                            print(f"  Added known: {actual_type} from {did}")
                except Exception as e:
                    print(f"  Known ID {did}: {e}")

        print(f"\nFound {len(type_samples)} dynamic types: {list(type_samples.keys())}")

        # Step 3: Render each type
        md_lines = [
            "# Bilibili Dynamic Render Test Report",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\nRendered {len(type_samples)} dynamic types.",
            "\n---\n",
        ]

        for dtype in sorted(type_samples.keys()):
            did, author, item = type_samples[dtype]
            desc = type_descriptions.get(dtype, dtype)
            page_url = f"https://t.bilibili.com/{did}"

            print(f"\nRendering {dtype} ({desc}) - ID: {did} - Author: {author}")

            try:
                img = await render_dynamic(engine, item)
                if img is None:
                    print(f"  FAILED: render returned None")
                    status = "❌ Render failed"
                    img_path = ""
                    w, h = 0, 0
                else:
                    img_path = path.join(out_dir, f"{dtype}_{did}.png")
                    img.save(img_path)
                    w, h = img.width(), img.height()
                    status = f"✅ {w}×{h}"
                    print(f"  OK: {w}x{h} saved to {img_path}")
            except Exception as e:
                print(f"  ERROR: {e}")
                status = f"❌ {e}"
                img_path = ""
                w, h = 0, 0

            # Analyze content
            modules = item.get("modules", {})
            dyn_mod = modules.get("module_dynamic", {}) or {}
            has_desc = bool(dyn_mod.get("desc"))
            major = dyn_mod.get("major", {}) or {}
            major_type = major.get("type", "none")
            has_major = bool(major)
            has_orig = "orig" in item
            has_forward = bool(item.get("forward"))

            features = []
            if has_desc: features.append("text")
            if has_major: features.append(f"major({major_type})")
            if has_orig: features.append("orig")
            if has_forward: features.append("forward")

            md_lines.append(f"## {desc} — `{dtype}`\n")
            md_lines.append(f"- **Author**: {author}")
            md_lines.append(f"- **ID**: `{did}`")
            md_lines.append(f"- **Source**: [🔗 {page_url}]({page_url})")
            md_lines.append(f"- **Features**: {', '.join(features) if features else 'none'}")
            md_lines.append(f"- **Status**: {status}")
            md_lines.append("")

            if img_path:
                rel_path = f"{dtype}_{did}.png"
                md_lines.append(f"[![{dtype}]({rel_path})]({page_url})")
            md_lines.append("\n---\n")

        # Write markdown
        md_file = path.join(out_dir, "report.md")
        with open(md_file, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        print(f"\n\nReport saved to {md_file}")
        print(f"Images saved to {out_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
