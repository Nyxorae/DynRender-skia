"""Profile rendering performance for a single Bilibili dynamic post.

Instruments every significant function in the render pipeline with wall-clock
timing, then prints a hierarchical breakdown showing where time is spent.

Usage:
    python profile_render.py [dynamic_id]
    python profile_render.py 1202655761342136353 --json
"""

import asyncio
import json
import sys
import time
from os import path

import httpx
import skia
from dynamicadaptor.DynamicConversion import formate_message

from dynrender_skia.config import create_style, init_static_path
from dynrender_skia.graphics import merge_pictures
from dynrender_skia.renderers import (
    BiliHeader,
    BiliRepost,
    BiliText,
    Footer,
    get_additional_renderer,
    get_major_renderer,
)

# ---------------------------------------------------------------------------
# Global profile store
# ---------------------------------------------------------------------------

_profiles: dict[str, dict] = {}


def _record(name: str, duration_ms: float, **extra):
    entry = {"name": name, "duration_ms": duration_ms}
    entry.update(extra)
    _profiles[name] = entry


# ---------------------------------------------------------------------------
# Monkey-patch helpers — wrap async methods with timing
# ---------------------------------------------------------------------------


def _patch_async(obj, method_name: str, profile_key: str):
    """Replace ``obj.method_name`` with a timed async wrapper."""
    original = getattr(obj, method_name)

    async def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            result = await original(*args, **kwargs)
        except Exception as exc:
            _profiles[profile_key] = {
                "name": profile_key,
                "duration_ms": (time.perf_counter() - t0) * 1000,
                "error": str(exc),
            }
            raise
        dt = (time.perf_counter() - t0) * 1000
        entry = {"name": profile_key, "duration_ms": dt}
        if result is not None:
            if hasattr(result, "shape"):
                entry["output_shape"] = list(result.shape)
            elif hasattr(result, "dimensions"):
                entry["output_size"] = (
                    f"{result.dimensions().width()}x{result.dimensions().height()}"
                )
        _profiles[profile_key] = entry
        return result

    setattr(obj, method_name, wrapper)


# ---------------------------------------------------------------------------
# Patched rendering pipeline
# ---------------------------------------------------------------------------


async def run_profiled(message, style, static_path, src_path):
    """Execute the full pipeline with fine-grained timing on every component."""

    # ------------------------------------------------------------------
    # 1. Header
    # ------------------------------------------------------------------
    header = BiliHeader(static_path, style)
    # Patch sub-methods
    _patch_async(header, "_paste_logo", "header.logo")
    _patch_async(header, "_draw_name", "header.name")
    _patch_async(header, "_draw_pub_time", "header.time")
    _patch_async(header, "_get_face_and_pendant", "header.fetch_face")
    _patch_async(header, "_past_face", "header.paste_face")
    _patch_async(header, "_paste_pendant", "header.paste_pendant")
    _patch_async(header, "_paste_vip", "header.paste_vip")

    # Wrap `run` to separate gather phase from sequential phase
    _patch_async(header, "run", "render.header")

    # ------------------------------------------------------------------
    # 2. Text
    # ------------------------------------------------------------------
    text_coro = None
    if message.text is not None:
        text_renderer = BiliText(static_path, style)
        _patch_async(text_renderer, "run", "render.text")
        # Patch internal steps
        _patch_async(text_renderer, "_make_topic", "text.topic")
        _patch_async(text_renderer, "_make_text_image", "text.body")
        _patch_async(text_renderer, "_get_emoji", "text.fetch_emoji")
        _patch_async(text_renderer, "_get_rich_pics", "text.fetch_icons")
        _patch_async(text_renderer, "_render_with_typesetter", "text.typeset_render")
        text_coro = text_renderer.run(message.text)

    # ------------------------------------------------------------------
    # 3. Major
    # ------------------------------------------------------------------
    major_coro = None
    if message.major is not None:
        cls = get_major_renderer(message.major.type)
        if cls:
            major_renderer = cls(src_path, style, message.major)
            _patch_async(major_renderer, "run", f"render.major({message.major.type})")

            # For OPUS type, patch sub-operations
            if message.major.type == "MAJOR_TYPE_OPUS":
                _patch_async(major_renderer, "_make_title", "major.opus.title")

            major_coro = major_renderer.run()

    # ------------------------------------------------------------------
    # 4. Forward
    # ------------------------------------------------------------------
    forward_coro = None
    if message.forward is not None:
        fwd = BiliRepost(static_path, style)
        _patch_async(fwd, "run", "render.forward")
        forward_coro = fwd.run(message.forward)

    # ------------------------------------------------------------------
    # 5. Additional
    # ------------------------------------------------------------------
    additional_coro = None
    if message.additional is not None:
        cls = get_additional_renderer(message.additional.type)
        if cls:
            add = cls(src_path, style, message.additional)
            _patch_async(add, "run", f"render.additional({message.additional.type})")
            additional_coro = add.run()

    # ------------------------------------------------------------------
    # 6. Footer
    # ------------------------------------------------------------------
    footer = Footer(static_path, style)
    _patch_async(footer, "run", "render.footer")

    # ------------------------------------------------------------------
    # Execute all render tasks concurrently
    # ------------------------------------------------------------------
    tasks_coros = [header.run(message.header)]
    task_names = ["header"]
    if text_coro:
        tasks_coros.append(text_coro)
        task_names.append("text")
    if major_coro:
        tasks_coros.append(major_coro)
        task_names.append(f"major({message.major.type})")
    if forward_coro:
        tasks_coros.append(forward_coro)
        task_names.append("forward")
    if additional_coro:
        tasks_coros.append(additional_coro)
        task_names.append(f"additional({message.additional.type})")
    tasks_coros.append(footer.run())
    task_names.append("footer")

    gather_t0 = time.perf_counter()
    section_results = await asyncio.gather(*tasks_coros)
    gather_ms = (time.perf_counter() - gather_t0) * 1000
    _record("render.gather_wall", gather_ms, note=f"concurrent window ({len(task_names)} tasks)")

    # ------------------------------------------------------------------
    # 7. Merge
    # ------------------------------------------------------------------
    merge_t0 = time.perf_counter()
    final = await merge_pictures(section_results)
    merge_ms = (time.perf_counter() - merge_t0) * 1000
    entry = {"name": "render.merge", "duration_ms": merge_ms}
    if final is not None:
        entry["output_shape"] = list(final.shape)
    _profiles["render.merge"] = entry

    return final


# ---------------------------------------------------------------------------
# Patch DynMajorDraw for image-fetch breakdown (used inside OPUS / DRAW)
# ---------------------------------------------------------------------------


def patch_all():
    """Apply monkey-patches for fine-grained profiling.

    Key insight: Python binds imported names at import time, so
    ``from .graphics import fetch_images`` creates a local reference.
    We must patch every module that uses the target function.
    """
    import dynrender_skia.graphics as gfx
    import dynrender_skia.renderers.major as major_mod
    import dynrender_skia.renderers.header as header_mod
    import dynrender_skia.renderers.text as text_mod

    _orig_fetch = gfx.fetch_images

    def _accumulate(key, ms, result=None):
        e = _profiles.get(key)
        if e is None:
            e = {"name": key, "duration_ms": 0, "calls": 0, "images_fetched": 0}
            _profiles[key] = e
        e["duration_ms"] += ms
        e["calls"] += 1
        if result is not None:
            if isinstance(result, (tuple, list)):
                e["images_fetched"] += len(result)
            else:
                e["images_fetched"] += 1

    async def patched_fetch(url, size=None, retries=5):
        t0 = time.perf_counter()
        try:
            result = await _orig_fetch(url, size, retries)
        except Exception:
            _accumulate("img.fetch_images", (time.perf_counter() - t0) * 1000)
            raise
        _accumulate("img.fetch_images", (time.perf_counter() - t0) * 1000, result)
        return result

    # Patch fetch_images everywhere it's imported
    gfx.fetch_images = patched_fetch
    major_mod.fetch_images = patched_fetch
    header_mod.fetch_images = patched_fetch
    text_mod.fetch_images = patched_fetch

    # ------------------------------------------------------------------
    # Patch DynMajorDraw methods for canvas-work breakdown
    # ------------------------------------------------------------------
    for method_name in ("_single_img", "_dual_img", "_triplex_img"):
        original = getattr(major_mod.DynMajorDraw, method_name)

        async def wrapper(self, *args, _orig=original, _name=method_name, **kwargs):
            t0 = time.perf_counter()
            try:
                result = await _orig(self, *args, **kwargs)
            except Exception as exc:
                _profiles[f"major.draw.{_name}"] = {
                    "name": f"major.draw.{_name}",
                    "duration_ms": (time.perf_counter() - t0) * 1000,
                    "error": str(exc),
                }
                raise
            dt = (time.perf_counter() - t0) * 1000
            entry = {"name": f"major.draw.{_name}", "duration_ms": dt}
            if result is not None and hasattr(result, "shape"):
                entry["output_shape"] = list(result.shape)
            _profiles[f"major.draw.{_name}"] = entry
            return result

        setattr(major_mod.DynMajorDraw, method_name, wrapper)

    # ------------------------------------------------------------------
    # Patch MajorOpus.run — time title / summary / pics separately
    # ------------------------------------------------------------------
    from dynrender_skia.renderers.major import MajorOpus
    _orig_opus_run = MajorOpus.run

    async def patched_opus_run(self, repost=False):
        # We time the three phases inline so we can attribute time correctly.
        # Phase A: title
        t0 = time.perf_counter()
        pics = []
        if self.major.opus.title:
            title_img = await self._make_title(self.major.opus.title, repost)
            pics.append(title_img)
        _record("major.opus.title", (time.perf_counter() - t0) * 1000)

        # Phase B: summary text
        t1 = time.perf_counter()
        if self.major.opus.summary:
            from dynamicadaptor.Content import RichTextDetail, Text
            from dynamicadaptor.Majors import RichTextNodes
            from dynrender_skia.renderers.text import BiliText
            from os import path as _path

            def _convert(node: RichTextNodes) -> RichTextDetail:
                return RichTextDetail(
                    type=node.type, text=node.text, orig_text=node.orig_text,
                    emoji=node.emoji.dict() if node.type == "RICH_TEXT_NODE_TYPE_EMOJI" else None,
                )
            dyn_text = Text(
                text=self.major.opus.summary.text, topic=None,
                rich_text_nodes=[
                    _convert(n) for n in self.major.opus.summary.rich_text_nodes
                ],
            )
            text_img = await BiliText(_path.dirname(self.src_path), self.style).run(dyn_text, repost)
            pics.append(text_img)
        _record("major.opus.summary_text", (time.perf_counter() - t1) * 1000)

        # Phase C: pictures
        t2 = time.perf_counter()
        if self.major.opus.pics:
            cover = await major_mod.DynMajorDraw(self.style, items=self.major.opus.pics).run(repost)
            pics.append(cover)
        _record("major.opus.pictures", (time.perf_counter() - t2) * 1000)

        if not pics:
            return None
        return await merge_pictures(pics)

    MajorOpus.run = patched_opus_run

    # ------------------------------------------------------------------
    # Patch BiliText sub-methods at class level (for OPUS summary timing)
    # ------------------------------------------------------------------
    _orig_make_text_image = text_mod.BiliText._make_text_image
    _orig_get_emoji = text_mod.BiliText._get_emoji
    _orig_render_typeset = text_mod.BiliText._render_with_typesetter

    async def patched_make_text_image(self, dyn_text):
        t0 = time.perf_counter()
        try:
            result = await _orig_make_text_image(self, dyn_text)
        except Exception as exc:
            _profiles["text.body"] = {"name": "text.body", "duration_ms": (time.perf_counter() - t0) * 1000, "error": str(exc)}
            raise
        _profiles["text.body"] = {"name": "text.body", "duration_ms": (time.perf_counter() - t0) * 1000}
        return result

    async def patched_get_emoji(self, emoji_urls, emoji_names):
        t0 = time.perf_counter()
        result = await _orig_get_emoji(self, emoji_urls, emoji_names)
        _profiles["text.fetch_emoji"] = {"name": "text.fetch_emoji", "duration_ms": (time.perf_counter() - t0) * 1000}
        return result

    async def patched_render_typeset(self, rich_list, dyn_text):
        t0 = time.perf_counter()
        result = await _orig_render_typeset(self, rich_list, dyn_text)
        _profiles["text.typeset_render"] = {"name": "text.typeset_render", "duration_ms": (time.perf_counter() - t0) * 1000}
        return result

    text_mod.BiliText._make_text_image = patched_make_text_image
    text_mod.BiliText._get_emoji = patched_get_emoji
    text_mod.BiliText._render_with_typesetter = patched_render_typeset


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_report(total_ms: float):
    print()
    print("=" * 72)
    print("  RENDER PERFORMANCE PROFILE")
    print("=" * 72)

    # Build display order
    sections = []

    # Phase 1: Pre-render
    sections.append(("─ Pre-render ─", None))
    for key, label in [
        ("1.api_fetch", "  API fetch (HTTP GET)"),
        ("2.format_message", "  Message format (dynamicadaptor)"),
    ]:
        if key in _profiles:
            sections.append((key, label))

    # Phase 2: Render pipeline
    sections.append(("─ Render pipeline ─", None))

    # 2a: Header detail
    sections.append(("render.header", "  HEADER (total)"))
    for key, label in [
        ("header.logo", "    ├─ paste_logo"),
        ("header.name", "    ├─ draw_name"),
        ("header.time", "    ├─ draw_pub_time"),
        ("header.fetch_face", "    ├─ fetch_face (avatar + pendant)"),
        ("header.paste_face", "    ├─ paste_face (circle crop + paste)"),
        ("header.paste_pendant", "    ├─ paste_pendant"),
        ("header.paste_vip", "    └─ paste_vip"),
    ]:
        if key in _profiles:
            sections.append((key, label))

    # 2b: Text detail
    sections.append(("render.text", "  TEXT (total)"))
    for key, label in [
        ("text.topic", "    ├─ make_topic"),
        ("text.fetch_emoji", "    ├─ fetch_emoji"),
        ("text.fetch_icons", "    ├─ fetch_rich_icons"),
        ("text.typeset_render", "    ├─ typeset + render"),
        ("text.body", "    └─ make_text_image (total)"),
    ]:
        if key in _profiles:
            sections.append((key, label))

    # 2c: Major detail
    for pk in sorted(_profiles):
        if pk.startswith("render.major("):
            sections.append((pk, f"  MAJOR {pk.split('(',1)[1].rstrip(')')} (total)"))
            break

    for key, label in [
        ("major.opus.title", "    ├─ make_title (title bar)"),
        ("major.opus.summary_text", "    ├─ summary text (BiliText + typeset)"),
        ("major.opus.pictures", "    ├─ pictures (DynMajorDraw.run)"),
        ("major.draw._triplex_img", "    │  └─ _triplex_img (3-pic layout)"),
        ("major.draw._dual_img", "    │  └─ _dual_img (2/4-pic layout)"),
        ("major.draw._single_img", "    │  └─ _single_img (1-pic layout)"),
    ]:
        if key in _profiles:
            sections.append((key, label))

    # 2d: Other sections
    for key, label in [
        ("render.forward", "  FORWARD / REPOST"),
        ("render.footer", "  FOOTER (timestamp)"),
    ]:
        if key in _profiles:
            sections.append((key, label))

    for key in sorted(_profiles):
        if key.startswith("render.additional("):
            sections.append((key, f"  ADDITIONAL {key.split('(',1)[1].rstrip(')')}"))
            break

    # Phase 3: Post-render
    sections.append(("─ Post-render ─", None))
    sections.append(("render.gather_wall", "  gather wall-clock (all tasks concurrent)"))
    sections.append(("render.merge", "  merge_pictures (np.concatenate)"))
    for key, label in [
        ("img.fetch_images", "  ├─ fetch_images (cross-cutting, all modules)"),
    ]:
        if key in _profiles:
            sections.append((key, label))

    print(f"\n{'Step':<48} {'ms':>8}  {'%':>6}")
    print("-" * 65)

    for key, label in sections:
        if label is None:
            print(f"  {key}")
            continue
        entry = _profiles.get(key)
        if entry is None:
            continue
        ms = entry["duration_ms"]
        pct = (ms / total_ms * 100) if total_ms > 0 else 0
        note = ""
        shape = entry.get("output_shape")
        if shape:
            note = f" → {shape[0]}x{shape[1]}"
        calls = entry.get("calls")
        if calls:
            note += f" ({calls} calls, {ms/calls:.0f}ms avg)"
        imgs = entry.get("images_fetched")
        if imgs:
            note += f" [{imgs} images]"
        err = entry.get("error")
        if err:
            note = f" ERROR: {err}"
        print(f"  {label:<46} {ms:>7.1f}  {pct:>5.1f}%{note}")

    print("-" * 65)
    print(f"  {'TOTAL':<46} {total_ms:>7.1f}  {'100.0%':>6}")
    print()

    # ---- Bottleneck analysis -----------------------------------------------
    # Find slowest operations (skip meta and aggregate entries)
    leaf_entries = [
        (k, v)
        for k, v in _profiles.items()
        if k != "meta"
        and not k.startswith("_")
        and k not in ("render.gather_wall", "render.merge")
    ]
    leaf_entries.sort(key=lambda x: x[1]["duration_ms"], reverse=True)

    if leaf_entries:
        print("Top 5 slowest operations:")
        for i, (k, v) in enumerate(leaf_entries[:5]):
            ms = v["duration_ms"]
            pct = (ms / total_ms * 100) if total_ms > 0 else 0
            print(f"  {i+1}. {k:<45} {ms:>7.1f} ms ({pct:.1f}%)")

    print()
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main(dynamic_id: str, json_output: bool = False):
    url = (
        "https://api.bilibili.com/x/polymer/web-dynamic/v1/detail"
        "?id={}&features=itemOpusStyle".format(dynamic_id)
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        ),
        "Referer": "https://t.bilibili.com/",
    }

    # ---- Phase 1: API fetch ----------------------------------------------
    t0 = time.perf_counter()
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers)
        data = resp.json()
    _record("1.api_fetch", (time.perf_counter() - t0) * 1000)

    if data["code"] != 0:
        print(f"API error: {data['message']}")
        return 1

    item = data["data"]["item"]
    dynamic_type = item.get("type", "unknown")

    _profiles["meta"] = {
        "dynamic_id": dynamic_id,
        "dynamic_type": dynamic_type,
    }

    print(f"Dynamic ID: {dynamic_id}")
    print(f"Type: {dynamic_type}")
    author = item.get("modules", {}).get("module_author", {}).get("name", "?")
    print(f"Author: {author}")

    # ---- Phase 2: Format message -----------------------------------------
    t0 = time.perf_counter()
    message = await formate_message(item)
    _record("2.format_message", (time.perf_counter() - t0) * 1000)

    if message is None:
        print("Failed to format message")
        return 1

    # Content summary
    sections = ["header"]
    if message.text:
        preview = (
            message.text.text[:40] if message.text.text else "(rich text only)"
        )
        sections.append(f"text: {preview}")
    if message.major:
        sections.append(f"major({message.major.type})")
        # Report opus sub-sections
        if message.major.type == "MAJOR_TYPE_OPUS":
            opus = message.major.opus
            bits = []
            if opus.title:
                bits.append("title")
            if opus.summary:
                bits.append("summary")
            if opus.pics:
                bits.append(f"pics({len(opus.pics)})")
            print(f"  OPUS content: {', '.join(bits)}")
    if message.forward:
        sections.append("forward")
    if message.additional:
        sections.append(f"additional({message.additional.type})")
    sections.append("footer")
    print(f"Pipeline: {' → '.join(sections)}")

    # ---- Phase 3: Render with profiling ----------------------------------
    from dynrender_skia.Core import DynRender

    render = DynRender()
    src_path = path.join(render.static_path, "Src")
    style = render.style

    # Apply all monkey-patches before rendering
    patch_all()

    total_t0 = time.perf_counter()
    img_array = await run_profiled(message, style, render.static_path, src_path)
    total_ms = (time.perf_counter() - total_t0) * 1000

    if img_array is None:
        print("Render returned None")
        return 1

    # ---- Save output -----------------------------------------------------
    output_path = f"profile_{dynamic_id}.png"
    img = skia.Image.fromarray(
        img_array, colorType=skia.ColorType.kRGBA_8888_ColorType
    )
    img.save(output_path)
    print(f"\nOutput: {output_path} ({img.width()}x{img.height()})")

    # ---- Report ----------------------------------------------------------
    if json_output:
        report = {
            "dynamic_id": dynamic_id,
            "dynamic_type": dynamic_type,
            "total_duration_ms": total_ms,
            "steps": {
                key: {
                    "label": entry.get("name", key),
                    "duration_ms": entry["duration_ms"],
                    "pct": (entry["duration_ms"] / total_ms * 100)
                    if total_ms > 0
                    else 0,
                }
                for key, entry in _profiles.items()
                if key != "meta"
            },
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(total_ms)

    return 0


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        sys.exit(0)

    json_out = "--json" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    did = args[0] if args else "1202655761342136353"
    sys.exit(asyncio.run(main(did, json_output=json_out)))
