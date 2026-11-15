import argparse
import time
import tracemalloc

from findex.storage import load_index_pickle
from findex.searcher import (
    search_merge, search_set, 
    search_merge_or, search_set_or, 
    search_merge_not, search_set_not
)

def main():
    parser = argparse.ArgumentParser(description="Пошук по індексу.")
    parser.add_argument("index_file", type=str, help="Шлях до файлу індексу")
    parser.add_argument("query", type=str, help='Запит у форматі term1 AND term2')
    parser.add_argument("--engine", choices=["merge", "set"], default="merge", help="Алгоритм пошуку")
    args = parser.parse_args()

    tracemalloc.start()
    start_time = time.perf_counter()

    inv_index, doc_meta = load_index_pickle(args.index_file)  
    load_time = time.perf_counter()

    parts = args.query.split()
    if len(parts) != 3:
        print("Помилка: запит має бути у форматі: term1 AND term2)")
        return

    term1, operator, term2 = parts[0].lower(), parts[1].upper(), parts[2].lower()

    if operator == "AND":
        func = search_merge if args.engine == "merge" else search_set
    elif operator == "OR":
        func = search_merge_or if args.engine == "merge" else search_set_or
    elif operator == "NOT":
        func = search_merge_not if args.engine == "merge" else search_set_not
    else:
        print(f"Невідомий оператор: {operator}")
        return

    doc_ids = func(inv_index, term1, term2)
    
    now, peak = tracemalloc.get_traced_memory()
    end_time = time.perf_counter()
    tracemalloc.stop()

    print(f"\n--- Результати для '{args.query}' (через {args.engine}) ---")
    print(f"Знайдено документів: {len(doc_ids)}")
    
    for d_id in doc_ids[:5]:
        filename = doc_meta[d_id].filename
        print(f" - {filename}")
        
    print("=" * 40)
    print(f"Час завантаження: {load_time - start_time:.4f} с")
    print(f"Час пошуку: {end_time - load_time:.6f} с")
    print(f"Пікове навантаження: {peak / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    main()