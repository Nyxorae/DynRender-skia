"""Benchmark rendering performance for all dynamic types.

Usage:
    python benchmark.py           # Run benchmarks, output to benchmark_output/report.md
"""

import asyncio
import os
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from os import path

import httpx
import skia
from dynamicadaptor.DynamicConversion import formate_message

from dynrender_skia.Core import DynRender
from dynrender_skia.graphics import merge_pictures
from dynrender_skia.renderers import (
    BiliHeader,
    BiliRepost,
    BiliText,
    Footer,
    get_additional_renderer,
    get_major_renderer,
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://t.bilibili.com/",
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

SECTIONS = ["header", "text", "major", "forward", "additional", "footer", "composite", "total"]


@dataclass
class R:
    tag: str = ""
    desc: str = ""
    did: str = ""
    api_type: str = ""
    w: int = 0
    h: int = 0
    total_s: float = 0
    times: dict = field(default_factory=dict)
    err: str = ""


async def _timed(name: str, coro):
    t0 = time.perf_counter()
    try:
        result = await coro
        dt = time.perf_counter() - t0
        return (name, dt, result)
    except Exception:
        dt = time.perf_counter() - t0
        return (name, dt, None)


async def render_one(engine: DynRender, msg) -> R:
    src = path.join(engine.static_path, "Src")
    t_total = time.perf_counter()
    tasks = []

    tasks.append(_timed("header", BiliHeader(engine.static_path, engine.style).run(msg.header)))
    if msg.text is not None:
        tasks.append(_timed("text", BiliText(engine.static_path, engine.style).run(msg.text)))
    if msg.major is not None:
        cls = get_major_renderer(msg.major.type)
        if cls:
            tasks.append(_timed("major", cls(src, engine.style, msg.major).run()))
    if msg.forward is not None:
        tasks.append(_timed("forward", BiliRepost(engine.static_path, engine.style).run(msg.forward)))
    if msg.additional is not None:
        cls = get_additional_renderer(msg.additional.type)
        if cls:
            tasks.append(_timed("additional", cls(src, engine.style, msg.additional).run()))
    tasks.append(_timed("footer", Footer(engine.static_path, engine.style).run()))

    items = await asyncio.gather(*tasks)

    valid = [v for _, _, v in items if v is not None]
    t_comp = time.perf_counter()
    img = None
    if valid:
        arr = await merge_pictures(valid)
        img = skia.Image.fromarray(arr, colorType=skia.ColorType.kRGBA_8888_ColorType)
    dt_comp = time.perf_counter() - t_comp
    dt_total = time.perf_counter() - t_total

    times = {n: d * 1000 for n, d, _ in items}
    times["composite"] = dt_comp * 1000
    times["total"] = dt_total * 1000

    return R(
        w=img.width() if img else 0, h=img.height() if img else 0,
        total_s=dt_total, times=times,
    )


async def fetch(client: httpx.AsyncClient, did: str):
    r = await client.get(DETAIL_URL.format(did), headers=HEADERS)
    data = r.json()
    return data if data.get("code") == 0 else None


def write_report(results: list[R], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    ok = [r for r in results if not r.err]
    totals = [r.total_s for r in ok]
    lines = [
        "# DynRender-skia Rendering Benchmark",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\nRenders: {len(ok)}/{len(results)} passed",
        f"\n## Per-Type Timing (ms)",
        "",
        f"| type | desc | size | API type | {' | '.join(SECTIONS)} | total |",
        f"|{'---' * (len(SECTIONS) + 4)}|",
    ]
    for r in ok:
        cells = [r.tag, r.desc, f"{r.w}x{r.h}", r.api_type]
        cells += [f"{r.times.get(s, 0):.0f}" for s in SECTIONS]
        cells.append(f"{r.total_s:.2f}s")
        lines.append("| " + " | ".join(cells) + " |")

    # Averages
    avg = ["**avg**", "", "", ""]
    for s in SECTIONS:
        avg.append(f"**{statistics.mean(r.times.get(s, 0) for r in ok):.0f}**")
    avg.append(f"**{statistics.mean(totals):.2f}s**")
    lines.append("| " + " | ".join(avg) + " |")

    lines += [
        "",
        "## Performance Summary",
        "",
        f"| metric | value |",
        f"|--------|-------|",
        f"| Renders | {len(ok)} passed, {len(results) - len(ok)} failed |",
        f"| Total time | {sum(totals):.2f}s |",
        f"| Min render | {min(totals):.2f}s |",
        f"| Max render | {max(totals):.2f}s |",
        f"| Avg render | {statistics.mean(totals):.2f}s |",
        f"| Median render | {statistics.median(totals):.2f}s |",
    ]

    # Per-section analysis
    lines += ["", "## Section Breakdown", ""]
    for s in SECTIONS:
        vals = [r.times.get(s, 0) for r in ok]
        if all(v == 0 for v in vals):
            lines.append(f"- **{s}**: N/A (not present in rendered types)")
        else:
            nonzero = [v for v in vals if v > 0]
            total_ms = sum(nonzero)
            lines.append(
                f"- **{s}**: total={total_ms:.0f}ms, avg={statistics.mean(nonzero):.0f}ms, "
                f"min={min(nonzero):.0f}ms, max={max(nonzero):.0f}ms ({len(nonzero)} renders)"
            )

    lines += [
        "",
        f"## Errors ({len(results) - len(ok)})",
        "",
    ]
    for r in results:
        if r.err:
            lines.append(f"- [{r.tag}] {r.desc} ({r.did}): `{r.err}`")

    path = os.path.join(out_dir, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


async def main():
    engine = DynRender()
    results: list[R] = []

    async with httpx.AsyncClient(timeout=30) as client:
        for tag, did, desc in TEST_IDS:
            print(f"  [{tag:>10s}] {desc:<10s} ", end="", flush=True)

            try:
                data = await fetch(client, did)
            except Exception as e:
                print(f"FAIL ({type(e).__name__})")
                results.append(R(tag=tag, desc=desc, did=did, err=f"Connection error: {e}"))
                continue
            if data is None:
                print("FAIL")
                results.append(R(tag=tag, desc=desc, did=did, err="API fetch failed"))
                continue

            msg = await formate_message(data["data"]["item"])
            if msg is None:
                print("FAIL")
                results.append(R(tag=tag, desc=desc, did=did, err="Parse failed"))
                continue

            api_type = data["data"]["item"].get("type", "?")
            r = await render_one(engine, msg)
            r.tag = tag
            r.desc = desc
            r.did = did
            r.api_type = api_type
            results.append(r)
            print(f"{r.total_s:.2f}s  {r.w}x{r.h}")

    path = write_report(results, "benchmark_output")
    print(f"\n  Report: {path}")


if __name__ == "__main__":
    asyncio.run(main())
