import argparse
import logging
import time
import tracemalloc

from findex.scorers import BM25, TfIdf
from findex.searcher import Searcher
from findex.storage import open_index


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    parser = argparse.ArgumentParser(description="Пошук по індексу.")
    parser.add_argument("index_file", type=str, help="Шлях до файлу індексу")
    parser.add_argument("query", type=str, help="Пошуковий запит")
    parser.add_argument("--scorer", choices=["bm25", "tfidf"], default="bm25", help="Алгоритм ранжування")
    parser.add_argument("--top-k", type=int, default=5, help="Кількість результатів")
    parser.add_argument("--no-snippets", action="store_true", help="Не генерувати сніпети")
    args = parser.parse_args()

    tracemalloc.start()
    start_time = time.perf_counter()

    scorer = BM25() if args.scorer == "bm25" else TfIdf()

    with open_index(args.index_file) as index:
        load_time = time.perf_counter()
        searcher = Searcher(index, scorer=scorer)
        results = searcher.search(args.query, top_k=args.top_k, with_snippets=not args.no_snippets)

    _, peak = tracemalloc.get_traced_memory()
    end_time = time.perf_counter()
    tracemalloc.stop()

    print(f"\n--- Результати для '{args.query}' ({args.scorer}) ---")
    print(f"Показано результатів: {len(results)}")

    for res in results:
        print(f"[{res.score:.4f}] {res.title} ({res.path})")
        if res.snippet:
            print(f"  {res.snippet}")

    print("=" * 40)
    print(f"Час завантаження: {load_time - start_time:.4f} с")
    print(f"Час пошуку: {end_time - load_time:.6f} с")
    print(f"Пікове навантаження: {peak / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()