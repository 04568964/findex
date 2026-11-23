import asyncio
import json
from pathlib import Path

import httpx

from findex.crawler import RobotsCache, canonicalize_url, crawl, fetch
from findex.indexer import build_index
from findex.iter_documents import read_file


def test_canonicalize_url() -> None:
    assert canonicalize_url("HTTP://Example.COM:80/a/b/?z=1&a=2#sec") == "http://example.com/a/b?a=2&z=1"
    assert canonicalize_url("/docs/", "https://example.com/index.html") == "https://example.com/docs"
    assert canonicalize_url("mailto:test@example.com") is None


def test_fetch_retries_429_and_respects_retry_after() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, text="ok", headers={"content-type": "text/html"})

    async def _run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            res = await fetch(client, "https://example.com/p", base_delay=0.01)
            assert res.status == 200
            assert calls == 2

    asyncio.run(_run())


def test_fetch_does_not_retry_404() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404)

    async def _run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            res = await fetch(client, "https://example.com/missing", base_delay=0.01)
            assert res.status == 404
            assert calls == 1

    asyncio.run(_run())


def test_robots_cache_caches_per_host() -> None:
    robots_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal robots_calls
        if request.url.path == "/robots.txt":
            robots_calls += 1
            return httpx.Response(200, text="User-agent: *\nDisallow: /private\n")
        return httpx.Response(200, text="ok")

    async def _run() -> None:
        cache = RobotsCache()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            assert await cache.can_fetch(client, "https://example.com/public") is True
            assert await cache.can_fetch(client, "https://example.com/private/x") is False
            assert robots_calls == 1

    asyncio.run(_run())


def test_crawl_and_jsonl_indexing(tmp_path: Path) -> None:
    pages_map = {
        "/": '<html><title>Home</title><body>Hello <a href="/a">A</a> <a href="/a#frag">A2</a> <a href="/b">B</a></body></html>',
        "/a": '<html><title>Page A</title><body>Python asyncio <a href="/c">C</a></body></html>',
        "/b": "<html><title>Page B</title><body>Search engine</body></html>",
        "/c": "<html><title>Page C</title><body>Extra page</body></html>",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        body = pages_map.get(request.url.path)
        if body is None:
            return httpx.Response(404)
        return httpx.Response(200, text=body, headers={"content-type": "text/html"})

    jsonl_path = tmp_path / "pages.jsonl"
    log_path = tmp_path / "crawl.log"

    async def _run() -> list[str]:
        urls: list[str] = []
        async for page in crawl(
            "https://example.com/",
            max_pages=3,
            concurrency=2,
            per_host_concurrency=2,
            host_delay=0.0,
            transport=httpx.MockTransport(handler),
            log_path=log_path,
        ):
            urls.append(page.url)
            f.write(json.dumps({"url": page.url, "title": page.title, "text": page.text}) + "\n")
        return urls

    with open(jsonl_path, "w", encoding="utf-8") as f:
        urls = asyncio.run(_run(), debug=True)

    assert len(urls) == 3
    assert len(set(urls)) == 3
    assert log_path.is_file()

    docs = list(read_file(jsonl_path))
    assert len(docs) == 3

    idx = build_index(jsonl_path, with_positions=True)
    assert idx.num_docs == 3
    assert "python" in idx or "hello" in idx