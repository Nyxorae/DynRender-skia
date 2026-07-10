import json

import aiofiles
import pytest
import skia
from dynamicadaptor.DynamicConversion import formate_message


def _has_font(family: str) -> bool:
    try:
        return skia.Typeface.MakeFromName(family, skia.FontStyle.Normal()) is not None
    except Exception:
        return False


@pytest.mark.skipif(not _has_font("Noto Sans SC"), reason="Requires Noto Sans SC font installed")
@pytest.mark.asyncio
async def test_dyn_render_run(resource_dir, dynrender_instance):
    async with aiofiles.open(resource_dir / "message.json", encoding="utf-8") as f:
        resp = await f.read()

    message_data = json.loads(resp)
    message = await formate_message("web", message_data["data"]["item"])
    img = await dynrender_instance.run(message)

    img = skia.Image.fromarray(img, colorType=skia.ColorType.kRGBA_8888_ColorType)
    img.save("preview.png")
