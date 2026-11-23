import argparse
import logging
import time
import tracemalloc

from findex.indexer import build_index
from findex.storage import save_index


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    parser = argparse.ArgumentParser(description="Будує інвертований індекс.")
    parser.add_argument("data", type=str, help="Шлях до папки з текстами")
    parser.add_argument("--out", type=str, required=True, help="Файл для збереження індексу")
    parser.add_argument("--positions", action="store_true", help="Зберігати позиції токенів для фраз")
    args = parser.parse_args()

    tracemalloc.start()
    start_time = time.perf_counter()

    index = build_index(args.data, with_positions=args.positions)
    save_index(index, args.out)

    _, peak = tracemalloc.get_traced_memory()
    end_time = time.perf_counter()
    tracemalloc.stop()

    print(f"Збережено у файл {args.out}")
    print(f"Збережено {len(index)} унікальних слів ({index.num_docs} документів)")
    print(f"Пікове навантаження: {peak / (1024 * 1024):.2f} MB")
    print(f"Час виконання: {end_time - start_time:.2f} с")


if __name__ == "__main__":
    main()