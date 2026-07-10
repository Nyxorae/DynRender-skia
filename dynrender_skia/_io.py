"""Image fetching — shared HTTP connection pool and request helpers."""

import asyncio
from typing import Optional, Union

import httpx
import skia
from loguru import logger

_IMG_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Referer": "https://t.bilibili.com/",
    "Origin": "https://t.bilibili.com",
}

_client: Optional[httpx.AsyncClient] = None
_client_lock = asyncio.Lock()


async def _get_client() -> httpx.AsyncClient:
    """Return the module-level shared HTTP client (lazy init, thread-safe)."""
    global _client
    if _client is not None and not _client.is_closed:
        return _client
    async with _client_lock:
        if _client is None or _client.is_closed:
            _client = httpx.AsyncClient(
                transport=httpx.AsyncHTTPTransport(retries=5),
                headers=_IMG_HEADERS,
                timeout=httpx.Timeout(30.0),
            )
        return _client


async def fetch_images(
    url: Union[str, list[str]], size: Optional[tuple[int, int]] = None, retries: int = 5
) -> Union[skia.Image, tuple[skia.Image, ...]]:
    """Fetch image(s) from URL(s), optionally resizing.

    Uses a shared HTTP client for connection pooling — orders of magnitude
    faster than creating a new client per call.
    """
    client = await _get_client()
    if isinstance(url, list):
        return await asyncio.gather(
            *[_request_img(client, u, size) for u in url]
        )
    return await _request_img(client, url, size)


async def _request_img(
    client: httpx.AsyncClient, url: str, size: Optional[tuple[int, int]],
) -> Optional[skia.Image]:
    try:
        response = await client.get(url)
        img: skia.Image = skia.Image.MakeFromEncoded(response.content)  # type: ignore
        if img is None:
            logger.error("Image decode error or request returned none in content")
        return img.resize(*size) if size is not None else img
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        logger.exception(f"Request or HTTP error occurred: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return None


request_img = _request_img
