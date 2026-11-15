import argparse
import time
import tracemalloc
from pathlib import Path
from findex.indexer import build_index, build_index_array_test
from findex.storage import save_index_pickle, save_index_json


def main():
    parser = argparse.ArgumentParser(description="Будує інвертований індекс.")
    parser.add_argument("data", type=str, help="Шлях до папки з текстами")
    parser.add_argument("--out", type=str, required=True, help="Файл для збереження індексу")
    args = parser.parse_args()

    tracemalloc.start()
    start_time = time.perf_counter()
    inv_index, doc_meta = build_index(args.data)
    save_index_json(inv_index, doc_meta, args.out)
    print(f"Збережено у файл {args.out}")

    now, peak = tracemalloc.get_traced_memory()
    end_time = time.perf_counter()
    tracemalloc.stop()

    print(f"Збережено {len(inv_index)} унікальних слів")
    print(f"Пікове навантаження: {'%.2f'%(peak/(1024*1024))} MB")
    print(f"Час виконання: {'%.2f'%(end_time-start_time)} с")

if __name__ == "__main__":
    main()