from pathlib import Path

from findex.scorers import BM25
from findex.searcher import Searcher
from findex.semantic import FastEmbedder, SemanticIndex, reciprocal_rank_fusion
from findex.storage import open_index

LAB3_QUERIES: list[tuple[str, set[int]]] = [
    ("american university", {420, 690}),
    ("state school", {656, 116, 882}),
    ("united states", {54, 158, 789, 282, 966, 195, 354, 626, 107, 217}),
    ("new york", {684, 612, 866, 795, 651}),
    ("first war", {90, 811, 318, 794, 434}),
    ("high school", {257, 508, 656, 86}),
    ("national park", {728, 773}),
    ("football team", {859, 784, 418, 312, 485, 444, 554, 371, 815, 677}),
    ("music album", {480, 767, 991, 601, 935, 314, 258, 242}),
    ("film director", {835, 281, 333, 10, 454, 727, 457, 276, 311}),
]

PARAPHRASED_QUERIES: list[tuple[str, set[int]]] = [
    ("higher education institution in the USA", {420, 690, 195, 107}),
    ("secondary educational establishment", {86, 252, 659, 854, 656, 257, 508}),
    ("federal republic in North America", {711, 354, 45, 107, 203}),
    ("armed conflict involving multiple nations", {794, 699, 512, 90, 434}),
    ("cinema filmmaker and movie production", {727, 311, 454, 40, 835, 333, 276}),
]


def precision_at_k(retrieved: list[int], relevant: set[int], k: int = 5) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return hits / k


def evaluate_group(
    name: str,
    queries: list[tuple[str, set[int]]],
    searcher: Searcher,
    sem_idx: SemanticIndex,
    embedder: FastEmbedder,
    num_docs: int,
) -> None:
    p_lex, p_sem, p_hyb = [], [], []
    for q, rel in queries:
        lex_hits = [r.doc_id for r in searcher.search(q, top_k=num_docs, with_snippets=False)]
        q_vec = embedder.embed_query(q)
        sem_hits = [doc_id for doc_id, _ in sem_idx.search(q_vec, top_k=num_docs)]
        hyb_hits = [
            doc_id for doc_id, _ in reciprocal_rank_fusion(lex_hits, sem_hits, k=60, top_k=5)
        ]

        p_lex.append(precision_at_k(lex_hits, rel, 5))
        p_sem.append(precision_at_k(sem_hits, rel, 5))
        p_hyb.append(precision_at_k(hyb_hits, rel, 5))

    n = len(queries) or 1
    print(
        f"| {name} ({len(queries)}) | {sum(p_lex) / n:.3f} | "
        f"{sum(p_sem) / n:.3f} | {sum(p_hyb) / n:.3f} |"
    )


def main() -> None:
    with open_index(Path("deploy/index.bin")) as idx:
        searcher = Searcher(idx, scorer=BM25(), use_numpy=True)
        sem_idx = SemanticIndex.load(Path("deploy/embeddings.npy"))
        embedder = FastEmbedder()

        print("| Query set | Lexical (BM25) P@5 | Semantic P@5 | Hybrid (RRF) P@5 |")
        print("| :--- | :---: | :---: | :---: |")
        evaluate_group("Lab 3 queries", LAB3_QUERIES, searcher, sem_idx, embedder, idx.num_docs)
        evaluate_group(
            "Paraphrased", PARAPHRASED_QUERIES, searcher, sem_idx, embedder, idx.num_docs
        )
        evaluate_group(
            "All queries",
            LAB3_QUERIES + PARAPHRASED_QUERIES,
            searcher,
            sem_idx,
            embedder,
            idx.num_docs,
        )


if __name__ == "__main__":
    main()