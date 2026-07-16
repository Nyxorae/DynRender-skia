"""Batch test: render specific Bilibili dynamic types using provided IDs."""
import asyncio
import os
from datetime import datetime
from os import path

import httpx
import skia
from dynamicadaptor.DynamicConversion import formate_message
from dynrender_skia.Core import DynRender

COOKIE = ""

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

TEST_IDS = [
    ("draw",       "741262186696933397", "图文/画册"),
    ("archive",    "739851131027456201", "视频投稿"),
    ("ugc_season", "755703296984875092", "合集季"),
    ("article",    "819930757423169558", "专栏文章"),
    ("common",     "551309621391003098", "通用卡片-1"),
    ("common",     "743181895357956118", "通用卡片-2"),
    ("music",      "819725994851041346", "音乐"),
    ("pgc",        "633983562923638785", "番剧/影视"),
    ("medialist",  "645144864359448578", "收藏列表"),
    ("courses",    "440646043801479846", "课程"),
    ("live",       "727260760787386403", "转发直播"),
]


async def fetch_dynamic(client: httpx.AsyncClient, did: str) -> dict | None:
    r = await client.get(DETAIL_URL.format(did), headers=HEADERS)
    data = r.json()
    if data.get("code") != 0:
        print(f"    API error: code={data.get('code')}, msg={data.get('message')}")
        return None
    return data


async def render_dynamic(engine: DynRender, item: dict) -> skia.Image | None:
    msg = await formate_message(item)
    if msg is None:
        return None
    arr = await engine.run(msg)
    if arr is None:
        return None
    return skia.Image.fromarray(arr, colorType=skia.ColorType.kRGBA_8888_ColorType)


async def main():
    engine = DynRender()
    out_dir = "test_output"
    os.makedirs(out_dir, exist_ok=True)

    md_lines = [
        "# Bilibili Dynamic Render Test Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\nRendering {len(TEST_IDS)} dynamics.",
        "\n---\n",
    ]

    async with httpx.AsyncClient(timeout=30) as client:
        for type_tag, did, desc in TEST_IDS:
            page_url = f"https://t.bilibili.com/{did}"
            print(f"\nRendering [{type_tag}] {desc} - ID: {did}")

            try:
                data = await fetch_dynamic(client, did)
                if data is None:
                    status = "❌ API fetch failed"
                    img_path = ""
                    w, h = 0, 0
                else:
                    item = data["data"]["item"]
                    actual_type = item.get("type", "unknown")
                    author = item.get("modules", {}).get("module_author", {}).get("name", "unknown")
                    print(f"    Author: {author}, API type: {actual_type}")

                    img = await render_dynamic(engine, item)
                    if img is None:
                        status = "❌ Render returned None"
                        img_path = ""
                        w, h = 0, 0
                    else:
                        img_path = path.join(out_dir, f"{type_tag}_{did}.png")
                        img.save(img_path)
                        w, h = img.width(), img.height()
                        status = f"✅ {w}×{h}"
                        print(f"    OK: {w}x{h} → {img_path}")
            except Exception as e:
                print(f"    ERROR: {e}")
                status = f"❌ {e}"
                img_path = ""
                w, h = 0, 0

            md_lines.append(f"## [{type_tag}] {desc}\n")
            md_lines.append(f"- **ID**: `{did}`")
            md_lines.append(f"- **Source**: [🔗 {page_url}]({page_url})")
            md_lines.append(f"- **Status**: {status}")
            md_lines.append("")

            if img_path:
                rel_path = f"{type_tag}_{did}.png"
                md_lines.append(f"![{type_tag}]({rel_path})")
            md_lines.append("\n---\n")

        md_file = path.join(out_dir, "report.md")
        with open(md_file, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        print(f"\n\nReport saved to {md_file}")
        print(f"Images saved to {out_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
