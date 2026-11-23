import asyncio
import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path

from findex.crawler import CrawlStats, crawl


def _get_peak_rss_mb() -> float:
    if sys.platform == "win32":

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return 0.0
        return counters.PeakWorkingSetSize / (1024 * 1024)

    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _run_cell(start_url: str, max_pages: int, concurrency: int, per_host: int, delay: float) -> dict[str, float | int]:
    stats = CrawlStats()
    out_path = Path(f"bench_c{concurrency}.jsonl")
    log_path = Path(f"bench_c{concurrency}.log")

    async def _go() -> None:
        async for page in crawl(
            start_url,
            max_pages=max_pages,
            concurrency=concurrency,
            per_host_concurrency=per_host,
            host_delay=delay,
            stats=stats,
            log_path=log_path,
        ):
            line = json.dumps({"url": page.url, "title": page.title, "text": page.text}, ensure_ascii=False) + "\n"
            await asyncio.to_thread(f.write, line)

    t0 = time.perf_counter()
    with open(out_path, "w", encoding="utf-8") as f:
        asyncio.run(_go())
    wall = time.perf_counter() - t0
    out_path.unlink(missing_ok=True)

    return {
        "wall": wall,
        "pages": stats.fetched,
        "pps": stats.fetched / wall if wall > 0 else 0.0,
        "errors": stats.errors,
        "rss": _get_peak_rss_mb(),
    }


def main() -> None:
    if "--cell" in sys.argv:
        i = sys.argv.index("--cell")
        res = _run_cell(
            start_url=sys.argv[i + 1],
            max_pages=int(sys.argv[i + 2]),
            concurrency=int(sys.argv[i + 3]),
            per_host=int(sys.argv[i + 4]),
            delay=float(sys.argv[i + 5]),
        )
        print(json.dumps(res))
        return

    start_url = sys.argv[1] if len(sys.argv) > 1 else "https://docs.python.org/3/"
    max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    delay = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05

    print(f"Start URL: {start_url} | max_pages={max_pages} | host_delay={delay}s\n")
    print("| Concurrency | Per-host | Wall (s) | Pages/s | Errors | Peak RSS (MB) |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: |")

    for c in (1, 5, 20):
        out = subprocess.check_output(
            [sys.executable, __file__, "--cell", start_url, str(max_pages), str(c), str(c), str(delay)],
            text=True,
        )
        m = json.loads(out.strip())
        print(
            f"| {c} | {c} | {m['wall']:.2f} | {m['pps']:.2f} | {m['errors']} | {m['rss']:.1f} |"
        )


if __name__ == "__main__":
    main()