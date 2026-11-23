import asyncio
import email.utils
import logging
import random
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx

USER_AGENT = "findex-crawler/0.6 (+mailto:8660119@stud.kai.edu.ua)"
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


@dataclass(slots=True)
class CrawlPage:
    url: str
    title: str
    text: str
    status: int
    size_bytes: int
    elapsed_s: float


@dataclass(slots=True)
class FetchResult:
    url: str
    status: int
    content: bytes
    content_type: str
    elapsed_s: float
    error: str | None = None


@dataclass(slots=True)
class CrawlStats:
    fetched: int = 0
    errors: int = 0
    in_flight: int = 0
    queue_size: int = 0
    started_at: float = field(default_factory=time.perf_counter)

    @property
    def pages_per_sec(self) -> float:
        elapsed = time.perf_counter() - self.started_at
        return self.fetched / elapsed if elapsed > 0 else 0.0


def canonicalize_url(url: str, base_url: str | None = None) -> str | None:
    resolved = urljoin(base_url, url) if base_url else url
    parts = urlsplit(resolved)
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https") or not parts.hostname:
        return None

    host = parts.hostname.lower()
    port = parts.port
    if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        port = None
    netloc = f"{host}:{port}" if port else host

    path = parts.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    query_items = sorted(parse_qsl(parts.query, keep_blank_values=True))
    query = urlencode(query_items)

    return urlunsplit((scheme, netloc, path, query, ""))


class _HTMLPageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._in_title = False
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t in ("script", "style", "noscript"):
            self._skip_depth += 1
        elif t == "title":
            self._in_title = True
        elif t == "a":
            for k, v in attrs:
                if k.lower() == "href" and v:
                    self.links.append(v)

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in ("script", "style", "noscript") and self._skip_depth > 0:
            self._skip_depth -= 1
        elif t == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        cleaned = data.strip()
        if not cleaned:
            return
        if self._in_title:
            self.title_parts.append(cleaned)
        self.text_parts.append(cleaned)


def parse_html(html: str, base_url: str) -> tuple[str, str, list[str]]:
    parser = _HTMLPageParser()
    parser.feed(html)
    title = " ".join(parser.title_parts).strip()
    text = " ".join(parser.text_parts).strip()
    links: list[str] = []
    for href in parser.links:
        canon = canonicalize_url(href, base_url)
        if canon:
            links.append(canon)
    return title, text, links


def _parse_retry_after(header_val: str | None) -> float | None:
    if not header_val:
        return None
    try:
        return max(0.0, float(header_val.strip()))
    except ValueError:
        pass
    try:
        dt = email.utils.parsedate_to_datetime(header_val)
        return max(0.0, dt.timestamp() - time.time())
    except (TypeError, ValueError):
        return None


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
    max_retries: int = 3,
    base_delay: float = 0.5,
) -> FetchResult:
    start = time.perf_counter()
    last_error: str | None = None
    last_status = 0

    for attempt in range(max_retries + 1):
        try:
            async with asyncio.timeout(timeout):
                resp = await client.get(url)
            last_status = resp.status_code

            if resp.status_code in RETRYABLE_STATUSES and attempt < max_retries:
                retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
                delay = (
                    retry_after
                    if retry_after is not None
                    else base_delay * (2**attempt) + random.uniform(0.0, base_delay)
                )
                await asyncio.sleep(delay)
                continue

            return FetchResult(
                url=str(resp.url),
                status=resp.status_code,
                content=resp.content,
                content_type=resp.headers.get("content-type", ""),
                elapsed_s=time.perf_counter() - start,
                error=None if resp.status_code < 400 else f"HTTP {resp.status_code}",
            )
        except (TimeoutError, httpx.TimeoutException, httpx.NetworkError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                delay = base_delay * (2**attempt) + random.uniform(0.0, base_delay)
                await asyncio.sleep(delay)
                continue
        except httpx.HTTPError as exc:
            return FetchResult(
                url=url,
                status=0,
                content=b"",
                content_type="",
                elapsed_s=time.perf_counter() - start,
                error=f"{type(exc).__name__}: {exc}",
            )

    return FetchResult(
        url=url,
        status=last_status,
        content=b"",
        content_type="",
        elapsed_s=time.perf_counter() - start,
        error=last_error or f"HTTP {last_status}",
    )


class RobotsCache:
    def __init__(self, user_agent: str = USER_AGENT) -> None:
        self._user_agent = user_agent
        self._parsers: dict[str, RobotFileParser] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def can_fetch(self, client: httpx.AsyncClient, url: str, timeout: float = 5.0) -> bool:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._locks:
            self._locks[origin] = asyncio.Lock()

        async with self._locks[origin]:
            if origin not in self._parsers:
                rp = RobotFileParser()
                robots_url = f"{origin}/robots.txt"
                try:
                    async with asyncio.timeout(timeout):
                        resp = await client.get(robots_url)
                    if resp.status_code == 200:
                        rp.parse(resp.text.splitlines())
                    else:
                        rp.parse([])
                except (TimeoutError, httpx.HTTPError):
                    rp.parse([])
                self._parsers[origin] = rp

        return self._parsers[origin].can_fetch(self._user_agent, url)


def _setup_crawl_logger(log_path: Path | None) -> tuple[logging.Logger, logging.FileHandler | None]:
    logger = logging.getLogger(f"findex.crawl.{id(log_path)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler: logging.FileHandler | None = None
    if log_path is not None:
        handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
    return logger, handler


async def crawl(
    start_url: str,
    *,
    max_pages: int = 200,
    concurrency: int = 5,
    per_host_concurrency: int = 2,
    host_delay: float = 0.2,
    same_host_only: bool = True,
    timeout: float = 10.0,
    max_retries: int = 3,
    base_delay: float = 0.5,
    transport: httpx.AsyncBaseTransport | None = None,
    stats: CrawlStats | None = None,
    log_path: Path | None = None,
) -> AsyncIterator[CrawlPage]:
    canon_start = canonicalize_url(start_url)
    if canon_start is None:
        raise ValueError(f"Invalid start URL: {start_url}")

    start_host = urlsplit(canon_start).netloc
    st = stats if stats is not None else CrawlStats()
    logger, file_handler = _setup_crawl_logger(log_path)

    url_queue: asyncio.Queue[str] = asyncio.Queue()
    out_queue: asyncio.Queue[CrawlPage | None] = asyncio.Queue()
    seen: set[str] = {canon_start}
    url_queue.put_nowait(canon_start)
    st.queue_size = 1

    global_sem = asyncio.Semaphore(max(1, concurrency))
    host_sems: dict[str, asyncio.Semaphore] = {}
    host_rate_locks: dict[str, asyncio.Lock] = {}
    host_last_req: dict[str, float] = {}
    robots = RobotsCache(USER_AGENT)
    stop_event = asyncio.Event()
    produced = 0

    async def _rate_limit(host: str) -> None:
        if host_delay <= 0:
            return
        if host not in host_rate_locks:
            host_rate_locks[host] = asyncio.Lock()
        async with host_rate_locks[host]:
            now = time.perf_counter()
            prev = host_last_req.get(host, 0.0)
            wait = host_delay - (now - prev)
            if wait > 0:
                await asyncio.sleep(wait)
            host_last_req[host] = time.perf_counter()

    async def _worker(client: httpx.AsyncClient) -> None:
        nonlocal produced
        while not stop_event.is_set():
            url = await url_queue.get()
            st.queue_size = url_queue.qsize()
            try:
                if stop_event.is_set():
                    return

                if not await robots.can_fetch(client, url, timeout=timeout):
                    logger.info("DISALLOWED %s", url)
                    continue

                host = urlsplit(url).netloc
                if host not in host_sems:
                    host_sems[host] = asyncio.Semaphore(max(1, per_host_concurrency))

                async with global_sem, host_sems[host]:
                    if stop_event.is_set():
                        return
                    await _rate_limit(host)
                    st.in_flight += 1
                    try:
                        res = await fetch(
                            client,
                            url,
                            timeout=timeout,
                            max_retries=max_retries,
                            base_delay=base_delay,
                        )
                    finally:
                        st.in_flight -= 1

                if res.error is not None or res.status >= 400:
                    st.errors += 1
                    logger.info("ERROR status=%d time=%.3fs url=%s err=%s", res.status, res.elapsed_s, url, res.error)
                    continue

                logger.info("OK status=%d bytes=%d time=%.3fs url=%s", res.status, len(res.content), res.elapsed_s, res.url)

                if "html" not in res.content_type.lower():
                    continue

                html = res.content.decode("utf-8", errors="replace")
                title, text, links = parse_html(html, res.url)

                if produced >= max_pages:
                    stop_event.set()
                    return

                produced += 1
                st.fetched = produced
                await out_queue.put(
                    CrawlPage(
                        url=res.url,
                        title=title or res.url,
                        text=text,
                        status=res.status,
                        size_bytes=len(res.content),
                        elapsed_s=res.elapsed_s,
                    )
                )

                if produced >= max_pages:
                    stop_event.set()
                    return

                for link in links:
                    if same_host_only and urlsplit(link).netloc != start_host:
                        continue
                    if link not in seen:
                        seen.add(link)
                        url_queue.put_nowait(link)
                st.queue_size = url_queue.qsize()
            finally:
                url_queue.task_done()
                st.queue_size = url_queue.qsize()

    async def _run_pool(client: httpx.AsyncClient) -> None:
        worker_tasks: list[asyncio.Task[None]] = []
        try:
            async with asyncio.TaskGroup() as tg:
                for _ in range(max(1, concurrency)):
                    worker_tasks.append(tg.create_task(_worker(client)))

                join_task = tg.create_task(url_queue.join())
                stop_task = tg.create_task(stop_event.wait())
                _done, pending = await asyncio.wait(
                    {join_task, stop_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                stop_event.set()
                for p in pending:
                    p.cancel()
                for wt in worker_tasks:
                    wt.cancel()
        except* asyncio.CancelledError:
            pass
        finally:
            await out_queue.put(None)

    try:
        async with httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
            max_redirects=5,
            transport=transport,
        ) as client:
            pool_task = asyncio.create_task(_run_pool(client))
            try:
                while True:
                    page = await out_queue.get()
                    if page is None:
                        break
                    yield page
            finally:
                stop_event.set()
                pool_task.cancel()
                try:
                    await pool_task
                except asyncio.CancelledError:
                    pass
    finally:
        if file_handler is not None:
            logger.removeHandler(file_handler)
            file_handler.close()