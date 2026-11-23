import ctypes
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from findex.indexer import ExecutorName, _split_chunks, build_partial, merge_indexes


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

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / 1024.0


def _run_once(
    all_paths: list[Path],
    workers: int,
    executor: ExecutorName,
) -> tuple[float, float, float]:
    chunks = _split_chunks(all_paths, workers)
    t0 = time.perf_counter()
    c0 = time.process_time()

    if executor == "serial" or len(chunks) == 1:
        partials = [build_partial(chunk, start_id, True) for start_id, chunk in chunks]
    else:
        pool_cls = ThreadPoolExecutor if executor == "threads" else ProcessPoolExecutor
        with pool_cls(max_workers=len(chunks)) as pool:
            futures = [pool.submit(build_partial, chunk, start_id, True) for start_id, chunk in chunks]
            partials = [f.result() for f in futures]

    m0 = time.perf_counter()
    _ = merge_indexes(partials)
    m1 = time.perf_counter()

    return m1 - t0, time.process_time() - c0, m1 - m0


def _bench_cell(executor: ExecutorName, workers: int, runs: int) -> dict[str, float]:
    all_paths = sorted(Path("data").glob("*.txt"))
    samples = [_run_once(all_paths, workers, executor) for _ in range(runs)]
    return {
        "wall": statistics.median(s[0] for s in samples),
        "cpu": statistics.median(s[1] for s in samples),
        "merge": statistics.median(s[2] for s in samples),
        "rss": _get_peak_rss_mb(),
    }


def main() -> None:
    if "--cell" in sys.argv:
        idx = sys.argv.index("--cell")
        ex: ExecutorName = sys.argv[idx + 1]  # type: ignore[assignment]
        w = int(sys.argv[idx + 2])
        runs = int(sys.argv[idx + 3])
        print(json.dumps(_bench_cell(ex, w, runs)))
        return

    all_paths = sorted(Path("data").glob("*.txt"))
    runs = 3
    cpu_count = os.cpu_count() or 1
    worker_counts = sorted({1, 2, 4, 8, cpu_count})
    gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)()

    # Прогрів кешу ОС перед замірами
    _run_once(all_paths, 1, "serial")

    print(f"Python: {sys.version.split()[0]} | sys._is_gil_enabled() = {gil_enabled}")
    print(f"CPU: {platform.processor()} | Logical cores: {cpu_count} | Docs: {len(all_paths)}\n")

    executors: list[ExecutorName] = ["serial", "threads", "processes"]
    if not gil_enabled:
        executors = ["serial", "threads"]

    baseline_wall: float | None = None
    results: dict[str, list[tuple[int, float]]] = {e: [] for e in executors}

    print("| Executor | Workers | Wall (s) | CPU (s) | Peak RSS (MB) | Merge (s) | Speedup |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for ex_name in executors:
        counts = [1] if ex_name == "serial" else worker_counts
        for w in counts:
            out = subprocess.check_output(
                [sys.executable, __file__, "--cell", ex_name, str(w), str(runs)],
                text=True,
            )
            m = json.loads(out.strip())
            if baseline_wall is None:
                baseline_wall = m["wall"]
            speedup = baseline_wall / m["wall"]
            results[ex_name].append((w, speedup))
            print(
                f"| {ex_name} | {w} | {m['wall']:.3f} | {m['cpu']:.3f} | "
                f"{m['rss']:.1f} | {m['merge']:.3f} | {speedup:.2f}x |"
            )

    if "--plot" in sys.argv:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        for ex_name, pts in results.items():
            if ex_name == "serial":
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            plt.plot(xs, ys, marker="o", label=ex_name)

        plt.plot(worker_counts, worker_counts, linestyle="--", color="gray", label="ideal linear")
        plt.xlabel("Workers")
        plt.ylabel("Speedup")
        plt.title(f"findex indexing speedup (GIL={gil_enabled})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("bench_speedup.png", dpi=150)
        print("\nSaved bench_speedup.png")


if __name__ == "__main__":
    main()