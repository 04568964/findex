import heapq
import logging
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from findex.models import Index, SearchResult
from findex.parser import parse_query
from findex.scorers import BM25, Scorer
from findex.snippets import build_snippet
from findex.timing import timed

logger = logging.getLogger("findex")


class Searcher:
    def __init__(
        self,
        index: Index,
        scorer: Scorer | None = None,
        use_numpy: bool = True,
    ) -> None:
        self.index = index
        self.scorer: Scorer = scorer or BM25()
        self.use_numpy = use_numpy
        self._resolve_doc_ids = lru_cache(maxsize=256)(self._resolve_doc_ids_uncached)

    def _resolve_doc_ids_uncached(self, query_str: str) -> frozenset[int]:
        logger.info("Cache miss for query: %r", query_str)
        ast = parse_query(query_str)
        return frozenset(ast.evaluate(self.index))

    @timed
    def search(
        self,
        query_str: str,
        top_k: int = 10,
        with_snippets: bool = True,
    ) -> list[SearchResult]:
        before = self._resolve_doc_ids.cache_info().hits
        candidate_ids = self._resolve_doc_ids(query_str)
        if self._resolve_doc_ids.cache_info().hits > before:
            logger.info("Cache hit for query: %r", query_str)

        if not candidate_ids:
            return []

        ast = parse_query(query_str)
        terms = ast.extract_terms()

        if self.use_numpy:
            doc_lengths = self.index.doc_lengths_np
            num_docs = self.index.num_docs
            avg_dl = self.index.avg_doc_length
            scores_arr = np.zeros(len(doc_lengths), dtype=np.float64)

            for term in set(terms):
                doc_ids, freqs = self.index.posting_arrays(term)
                if len(doc_ids) == 0:
                    continue
                term_scores = self.scorer.score_array(
                    doc_ids, freqs, doc_lengths, num_docs, avg_dl
                )
                scores_arr[doc_ids] += term_scores

            cand_ids = np.fromiter(candidate_ids, dtype=np.int32, count=len(candidate_ids))
            cand_scores = scores_arr[cand_ids]

            k = min(top_k, len(cand_ids))
            if k < len(cand_ids):
                top_idx = np.argpartition(-cand_scores, k - 1)[:k]
                top_idx = top_idx[np.lexsort((cand_ids[top_idx], -cand_scores[top_idx]))]
            else:
                top_idx = np.lexsort((cand_ids, -cand_scores))

            top_items = [(int(cand_ids[i]), float(cand_scores[i])) for i in top_idx]
        else:
            scores: dict[int, float] = defaultdict(float)
            for term in set(terms):
                if term not in self.index:
                    continue
                for posting in self.index[term]:
                    if posting.doc_id in candidate_ids:
                        scores[posting.doc_id] += self.scorer.score(term, posting, self.index)

            for doc_id in candidate_ids:
                scores.setdefault(doc_id, 0.0)

            top_items = heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])

        results: list[SearchResult] = []
        for doc_id, score in top_items:
            meta = self.index.docs[doc_id]
            snippet = ""
            if with_snippets and Path(meta.path).is_file():
                text = Path(meta.path).read_text(encoding="utf-8", errors="ignore")
                snippet = build_snippet(text, terms)
            results.append(
                SearchResult(
                    score=score,
                    doc_id=doc_id,
                    title=meta.title,
                    path=meta.path,
                    snippet=snippet,
                )
            )
        return results