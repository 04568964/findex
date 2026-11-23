import sys

from findex.scorers import BM25, TfIdf
from findex.searcher import Searcher
from findex.storage import open_index

QUERIES = [
    "university",
    "american AND civil",
    "world AND war",
    "united AND states",
    "music AND album",
    "football AND league",
    "film AND director",
    "science AND research",
    "history AND century",
    "political AND party",
]

RELEVANT: dict[str, set[str]] = {
    "university": {"doc_0442", "doc_0094", "doc_0892", "doc_0896", "doc_0746"},
    "american AND civil": {"doc_0421", "doc_0983", "doc_0107", "doc_0045", "doc_0195"},
    "world AND war": {"doc_0794", "doc_0610", "doc_0237", "doc_0090", "doc_0280"},
    "united AND states": {"doc_0054", "doc_0779", "doc_0816", "doc_0158", "doc_0789", "doc_0282"},
    "music AND album": {"doc_0981", "doc_0148", "doc_0242", "doc_0480", "doc_0767"},
    "football AND league": {"doc_0769", "doc_0444", "doc_0786", "doc_0352", "doc_0417"},
    "film AND director": {"doc_0810", "doc_0267", "doc_0281", "doc_0333", "doc_0010"},
    "science AND research": {"doc_0048", "doc_0099", "doc_0136", "doc_0028"},
    "history AND century": {"doc_0913", "doc_0345", "doc_0634", "doc_0879", "doc_0221"},
    "political AND party": {"doc_0215", "doc_0600", "doc_0168", "doc_0410", "doc_0134"},
}


def precision_at_5(results, relevant: set[str]) -> float:
    if not results:
        return 0.0
    hits = sum(1 for r in results[:5] if r.title in relevant)
    return hits / 5.0


def main() -> None:
    inspect = "--inspect" in sys.argv
    index_path = "index.bin"

    with open_index(index_path) as index:
        s_tfidf = Searcher(index, TfIdf())
        s_bm25 = Searcher(index, BM25())

        if inspect:
            for q in QUERIES:
                print(f"\n=== {q} ===")
                t_res = s_tfidf.search(q, top_k=5, with_snippets=True)
                b_res = s_bm25.search(q, top_k=5, with_snippets=True)
                seen = {}
                for r in t_res + b_res:
                    seen.setdefault(r.title, r.snippet)
                for title, snip in seen.items():
                    print(f"  {title}: {snip[:120]}")
            return

        print("| Запит | TF-IDF P@5 | BM25 P@5 |")
        print("| :--- | :---: | :---: |")
        t_scores, b_scores = [], []
        for q in QUERIES:
            rel = RELEVANT[q]
            p_t = precision_at_5(s_tfidf.search(q, top_k=5, with_snippets=False), rel)
            p_b = precision_at_5(s_bm25.search(q, top_k=5, with_snippets=False), rel)
            t_scores.append(p_t)
            b_scores.append(p_b)
            print(f"| `{q}` | {p_t:.2f} | {p_b:.2f} |")
        print(f"| **Середнє** | **{sum(t_scores)/len(t_scores):.2f}** | **{sum(b_scores)/len(b_scores):.2f}** |")


if __name__ == "__main__":
    main()