import sys

from findex.models import Posting
from findex.scorers import BM25
from findex.storage import open_index


def main() -> None:
    index_path = sys.argv[1] if len(sys.argv) > 1 else "index.bin"
    bm25 = BM25(k1=1.5, b=0.75)

    with open_index(index_path) as index:
        print(f"Index: {index}, avg_doc_length={index.avg_doc_length:.2f}\n")
        rare_term = "murderous"
        common_term = "the"
        rare_p = index[rare_term][0]
        doc_id = rare_p.doc_id
        s_rare = bm25.score(rare_term, Posting(doc_id, 1), index)
        s_common = bm25.score(common_term, Posting(doc_id, 1), index)
        print("Порівняння рідкісної та частого терміну (однаковий doc_id, tf=1):")
        print(f"'{rare_term}' (df={index.df(rare_term)}): {s_rare:.4f}")
        print(f"'{common_term}' (df={index.df(common_term)}): {s_common:.4f}\n")
        term = "university"
        s1 = bm25.score(term, Posting(doc_id, 1), index)
        s19 = bm25.score(term, Posting(doc_id, 19), index)
        s20 = bm25.score(term, Posting(doc_id, 20), index)
        print(f"Насичення TF для '{term}' (df={index.df(term)}):")
        print(f"tf=1:  {s1:.4f}")
        print(f"tf=19: {s19:.4f}")
        print(f"tf=20: {s20:.4f} (приріст: +{s20 - s19:.6f})\n")
        single_occ = [p for p in index[term] if p.freq == 1]
        short_p = min(single_occ, key=lambda p: index.docs[p.doc_id].length)
        long_p = max(single_occ, key=lambda p: index.docs[p.doc_id].length)
        s_short = bm25.score(term, short_p, index)
        s_long = bm25.score(term, long_p, index)
        print(f"Нормалізація довжини для '{term}' (tf=1):")
        print(f"Короткий (doc={index.docs[short_p.doc_id].title}, len={index.docs[short_p.doc_id].length}): {s_short:.4f}")
        print(f"Довгий (doc={index.docs[long_p.doc_id].title}, len={index.docs[long_p.doc_id].length}): {s_long:.4f}")


if __name__ == "__main__":
    main()