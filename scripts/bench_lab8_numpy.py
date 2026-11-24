import statistics
import time
from pathlib import Path

from findex.scorers import BM25
from findex.searcher import Searcher
from findex.storage import open_index


def measure_ms(searcher: Searcher, query: str, repeats: int = 300) -> float:
    searcher.search(query, top_k=10, with_snippets=False)
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        searcher.search(query, top_k=10, with_snippets=False)
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def main() -> None:
    with open_index(Path("deploy/index.bin")) as idx:
        by_df = sorted(((len(v), k) for k, v in idx.items()), reverse=True)
        frequent = by_df[0][1]
        rare = next(k for df, k in reversed(by_df) if df >= 2)
        three_terms = "state school american"

        py_searcher = Searcher(idx, scorer=BM25(), use_numpy=False)
        np_searcher = Searcher(idx, scorer=BM25(), use_numpy=True)

        print("| Query type | Query | Lab 3 (ms) | NumPy (ms) | Speedup |")
        print("| :--- | :--- | :---: | :---: | :---: |")
        for label, q in [
            ("Frequent term", frequent),
            ("Rare term", rare),
            ("3-term query", three_terms),
        ]:
            t_py = measure_ms(py_searcher, q)
            t_np = measure_ms(np_searcher, q)
            speedup = t_py / t_np if t_np > 0 else 0.0
            print(f"| {label} | `{q}` | {t_py:.4f} | {t_np:.4f} | {speedup:.2f}x |")


if __name__ == "__main__":
    main()