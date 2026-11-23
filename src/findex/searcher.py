import heapq
import logging
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from findex.models import Index, SearchResult
from findex.parser import parse_query
from findex.scorers import BM25, Scorer
from findex.snippets import build_snippet
from findex.timing import timed

logger = logging.getLogger("findex")


class Searcher:
    def __init__(self, index: Index, scorer: Scorer | None = None) -> None:
        self.index = index
        self.scorer: Scorer = scorer or BM25()
        self._resolve_doc_ids = lru_cache(maxsize=256)(self._resolve_doc_ids_uncached)

    def _resolve_doc_ids_uncached(self, query_str: str) -> frozenset[int]:
        logger.info("Cache miss for query: %r", query_str)
        ast = parse_query(query_str)
        return frozenset(ast.evaluate(self.index))

    @timed
    def search(self, query_str: str, top_k: int = 10, with_snippets: bool = True) -> list[SearchResult]:
        before = self._resolve_doc_ids.cache_info().hits
        candidate_ids = self._resolve_doc_ids(query_str)
        if self._resolve_doc_ids.cache_info().hits > before:
            logger.info("Cache hit for query: %r", query_str)

        if not candidate_ids:
            return []

        ast = parse_query(query_str)
        terms = ast.extract_terms()
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
