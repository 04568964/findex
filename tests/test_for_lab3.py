import pytest
from findex.models import DocMeta, Index, Posting
from findex.parser import And, Not, Or, Phrase, Term, parse_query
from findex.scorers import BM25, TfIdf
from findex.searcher import Searcher
from findex.storage import open_index, save_index


def test_parser_complex_ast() -> None:
    ast = parse_query('python AND (async OR await) NOT java "event loop"')
    expected = (
        Term("python")
        & (Term("async") | Term("await"))
        & ~Term("java")
        & Phrase(("event", "loop"))
    )
    assert ast == expected


def test_parser_precedence() -> None:
    assert parse_query("a OR b AND c") == Or(Term("a"), And(Term("b"), Term("c")))
    assert parse_query("NOT a AND b") == And(Not(Term("a")), Term("b"))


def test_index_mapping_and_context_manager(tmp_path) -> None:
    idx = Index(
        postings={"python": [Posting(0, 2, (0, 3))]},
        docs={0: DocMeta(0, "a.txt", "a", 10)},
        has_positions=True,
    )
    assert len(idx) == 1
    assert "python" in idx
    assert idx["python"][0].freq == 2
    assert list(idx) == ["python"]
    assert idx.num_docs == 1
    assert idx.avg_doc_length == 10.0
    assert idx.df("python") == 1

    out = tmp_path / "idx.bin"
    save_index(idx, out)

    with pytest.raises(RuntimeError):
        with open_index(out) as loaded:
            assert "python" in loaded
            raise RuntimeError("boom")
    assert len(loaded) == 0


def test_bm25_properties() -> None:
    docs = {
        0: DocMeta(0, "d0", "short", 10),
        1: DocMeta(1, "d1", "long", 500),
        2: DocMeta(2, "d2", "saturated", 100),
        3: DocMeta(3, "d3", "filler", 100),
    }
    postings = {
        "common": [Posting(0, 1), Posting(1, 1), Posting(2, 1), Posting(3, 1)],
        "rare": [Posting(0, 1)],
        "repeat": [Posting(2, 19), Posting(3, 20)],
    }
    idx = Index(postings, docs)
    bm25 = BM25()

    assert bm25.score("rare", Posting(0, 1), idx) > bm25.score("common", Posting(0, 1), idx)

    s19 = bm25.score("repeat", Posting(2, 19), idx)
    s20 = bm25.score("repeat", Posting(3, 20), idx)
    assert (s20 - s19) < 0.02

    s_short = bm25.score("common", Posting(0, 1), idx)
    s_long = bm25.score("common", Posting(1, 1), idx)
    assert s_short > s_long


def test_phrase_and_cache() -> None:
    docs = {
        0: DocMeta(0, "d0", "d0", 5),
        1: DocMeta(1, "d1", "d1", 5),
    }
    postings = {
        "event": [Posting(0, 1, (0,)), Posting(1, 1, (0,))],
        "loop": [Posting(0, 1, (1,)), Posting(1, 1, (3,))],
    }
    idx = Index(postings, docs, has_positions=True)
    searcher = Searcher(idx, TfIdf())

    res1 = searcher.search('"event loop"', with_snippets=False)
    res2 = searcher.search('"event loop"', with_snippets=False)
    assert [r.doc_id for r in res1] == [0]
    assert [r.doc_id for r in res2] == [0]
    assert searcher._resolve_doc_ids.cache_info().hits == 1