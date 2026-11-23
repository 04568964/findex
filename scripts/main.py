import time
import tracemalloc
from collections import Counter
from findex.iter_documents import read_file, greedy_read_files
from findex.tokenizer import tokenizer
from pathlib import Path




#TEMP
from operator import itemgetter
import heapq
def least_common_values(array, to_find=None):
    counter = Counter(array)
    if to_find is None:
        return sorted(counter.items(), key=itemgetter(1), reverse=False)
    return heapq.nsmallest(to_find, counter.items(), key=itemgetter(1))
#TEMP


def main():
    data_path = Path("data")

    tracemalloc.start()
    l_start_time = time.perf_counter()
    doc_counter = 0
    total_tokens_count = 0
    vocab = Counter()

    for filename, text in read_file(data_path):
        doc_counter +=1
        for token in tokenizer(text):
            total_tokens_count +=1
            vocab[token] += 1
    now, l_peak = tracemalloc.get_traced_memory()
    l_end_time = time.perf_counter()
    tracemalloc.stop()

    tracemalloc.start()
    g_start_time = time.perf_counter()
    doc_counter = 0
    total_tokens_count = 0
    vocab = Counter()

    for text in greedy_read_files(data_path):
        doc_counter +=1
        for token in tokenizer(text):
            total_tokens_count +=1
            vocab[token] += 1
    now, g_peak = tracemalloc.get_traced_memory()
    g_end_time = time.perf_counter()
    tracemalloc.stop()

    print("="*60)
    print(f"Лінива версія")
    print(f"Пікове навантаження: {'%.2f'%(l_peak/(1024*1024))} MB")
    print(f"Час виконання: {'%.2f'%(l_end_time-l_start_time)} с")
    print("="*60)
    print(f"Жадібна версія")
    print(f"Пікове навантаження: {'%.2f'%(g_peak/(1024*1024))} MB")
    print(f"Час виконання: {'%.2f'%(g_end_time-g_start_time)} с")
    print("="*60)
    print("Статистика")
    print(f"Оброблено документів: {doc_counter}")
    print(f"Загальна кількість токенів: {total_tokens_count}")
    print(f"Унікальних слів: {len(vocab)}")

    for word, count in least_common_values(vocab,2):
        print(f"{word} : {count}")

        


if __name__ == "__main__":
    main()