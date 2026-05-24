"""Generate a test preview image for a given dynamic ID."""
import asyncio
import json
import sys

import httpx
import skia
from dynamicadaptor.DynamicConversion import formate_message
from dynrender_skia.Core import DynRender


async def main(dynamic_id: str):
    url = 'https://api.bilibili.com/x/polymer/web-dynamic/v1/detail?id={}&features=itemOpusStyle'.format(dynamic_id)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://t.bilibili.com/",
        "cookie": "",  # Set your Bilibili cookie here if needed
    }

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers)
        data = resp.json()

    if data["code"] != 0:
        print(f"API error: {data['message']}")
        return 1

    # Save raw API response
    # with open(f"dynamic_{dynamic_id}.json", "w", encoding="utf-8") as f:
    #     json.dump(data, f, ensure_ascii=False, indent=2)
    # print(f"Raw JSON saved to dynamic_{dynamic_id}.json")

    item = data["data"]["item"]
    print(f"Dynamic type: {item.get('type')}")
    print(f"Author: {item['modules']['module_author']['name']}")

    # Convert web response to RenderMessage
    message = await formate_message("web", item)
    if message is None:
        print("Failed to formate message")
        return 1

    # Render
    render = DynRender()
    img_array = await render.run(message)

    if img_array is None:
        print("Failed to render image")
        return 1

    # Save
    output_path = f"preview_{dynamic_id}.png"
    img = skia.Image.fromarray(img_array, colorType=skia.ColorType.kRGBA_8888_ColorType)
    img.save(output_path)
    print(f"Saved to {output_path} ({img.width()}x{img.height()})")
    return 0


if __name__ == "__main__":
    did = sys.argv[1] if len(sys.argv) > 1 else "1202655761342136353"
    sys.exit(asyncio.run(main(did)))
