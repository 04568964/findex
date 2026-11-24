import cProfile
import pstats
import sys
from pathlib import Path

from findex.indexer import build_index
from findex.scorers import BM25
from findex.searcher import Searcher
from findex.storage import open_index


def profile_index(data_dir: Path) -> None:
    pr = cProfile.Profile()
    pr.enable()
    build_index(data_dir, with_positions=True, executor="serial")
    pr.disable()
    print("=== cProfile: build_index (top 15 cumtime) ===")
    pstats.Stats(pr).sort_stats("cumtime").print_stats(15)


def profile_search(index_path: Path, query: str, repeats: int = 200) -> None:
    with open_index(index_path) as idx:
        searcher = Searcher(idx, scorer=BM25())
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(repeats):
            searcher.search(query, top_k=10, with_snippets=False)
        pr.disable()
    print(f"=== cProfile: search('{query}') x{repeats} (top 15 cumtime) ===")
    pstats.Stats(pr).sort_stats("cumtime").print_stats(15)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    data_dir = Path("data")
    index_path = Path("deploy/index.bin")
    query = "state school american"

    if mode in ("both", "index"):
        profile_index(data_dir)
    if mode in ("both", "search"):
        profile_search(index_path, query)