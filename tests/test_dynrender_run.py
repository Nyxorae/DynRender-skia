import json
import os

import aiofiles
import pytest
import skia
from dynamicadaptor.DynamicConversion import formate_message


@pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip this test in CI environments.")
@pytest.mark.asyncio
async def test_dyn_render_run(shared_cache, resource_dir, dynrender_instance):
    async with aiofiles.open(resource_dir / "message.json", encoding="utf-8") as f:
        resp = await f.read()

    message_data = json.loads(resp)
    message = await formate_message("web", message_data["data"]["item"])
    img = await dynrender_instance.run(message)

    img = skia.Image.fromarray(img, colorType=skia.ColorType.kRGBA_8888_ColorType)
    img.save("preview.png")
