from pathlib import Path

import numpy as np
import pytest

from findex.models import DocMeta, Index, Posting
from findex.scorers import BM25, TfIdf
from findex.searcher import Searcher
from findex.semantic import SemanticIndex, chunk_text, reciprocal_rank_fusion


def test_chunk_text_overlap() -> None:
    text = " ".join(f"w{i}" for i in range(25))
    chunks = chunk_text(text, chunk_words=10, overlap=2)
    assert len(chunks) == 3
    assert chunks[0].split()[-2:] == chunks[1].split()[:2]

def test_semantic_index_best_chunk_and_save_load(tmp_path: Path) -> None:
    vecs = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.6],
        ],
        dtype=np.float32,
    )
    doc_ids = np.array([0, 0, 1], dtype=np.int32)
    sem = SemanticIndex(vectors=vecs, chunk_doc_ids=doc_ids)

    out = tmp_path / "emb.npy"
    sem.save(out)
    loaded = SemanticIndex.load(out)

    q = np.array([1.0, 0.0], dtype=np.float32)
    res = loaded.search(q, top_k=2)
    assert [d for d, _ in res] == [0, 1]
    assert res[0][1] == pytest.approx(1.0)
    assert res[1][1] == pytest.approx(0.8)


def test_rrf_combines_rankings() -> None:
    fused = reciprocal_rank_fusion([10, 20, 30], [30, 10, 40], k=60, top_k=4)
    assert {fused[0][0], fused[1][0]} == {10, 30}

def _sample_index() -> Index:
    docs = {
        0: DocMeta(doc_id=0, path="d0.txt", title="D0", length=120),
        1: DocMeta(doc_id=1, path="d1.txt", title="D1", length=80),
        2: DocMeta(doc_id=2, path="d2.txt", title="D2", length=200),
        3: DocMeta(doc_id=3, path="d3.txt", title="D3", length=50),
    }
    postings = {
        "state": [Posting(0, 3), Posting(1, 1), Posting(2, 5), Posting(3, 2)],
        "school": [Posting(0, 1), Posting(2, 2)],
        "american": [Posting(1, 4), Posting(3, 1)],
    }
    return Index(postings, docs, has_positions=False)


@pytest.mark.parametrize("scorer_cls", [BM25, TfIdf])
@pytest.mark.parametrize("query", ["state", "school", "state school american"])
def test_numpy_matches_lab3_order(scorer_cls: type[BM25] | type[TfIdf], query: str) -> None:
    idx = _sample_index()
    py_res = Searcher(idx, scorer=scorer_cls(), use_numpy=False).search(query, top_k=10, with_snippets=False)
    np_res = Searcher(idx, scorer=scorer_cls(), use_numpy=True).search(query, top_k=10, with_snippets=False)

    assert [r.doc_id for r in np_res] == [r.doc_id for r in py_res]
    for r_np, r_py in zip(np_res, py_res, strict=True):
        assert r_np.score == pytest.approx(r_py.score, rel=1e-9)