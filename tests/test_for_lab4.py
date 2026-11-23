import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from typer.testing import CliRunner

from findex.boolean import merge_and, merge_not, merge_or
from findex.cli import app
from findex.models import DocMeta, Index, Posting
from findex.parser import And, Not, Or, Phrase, Term, parse_query
from findex.scorers import BM25, Scorer, TfIdf
from findex.storage import load_index, open_index, save_index
from findex.tokenizer import tokenizer

runner = CliRunner()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Hello, World!", ["hello", "world"]),
        ("don’t stop", ["don't", "stop"]),
        ("well—known fact", ["well-known", "fact"]),
        ("Python 3.13", ["python", "3", "13"]),
        ("café résumé", ["caf", "r", "sum"]),
        ("   \n\t  ", []),
        ("A-B-C", ["a-b-c"]),
        ("ﬁle ﬂow", ["file", "flow"]),
    ],
)
def test_tokenizer_cases(raw: str, expected: list[str]) -> None:
    assert list(tokenizer(raw)) == expected


def test_build_index_from_corpus(sample_index: Index) -> None:
    assert sample_index.num_docs == 5
    assert "python" in sample_index
    assert sample_index.df("python") == 4
    assert sample_index.has_positions is True


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("python", Term("python")),
        ("a AND b", And(Term("a"), Term("b"))),
        ("a OR b", Or(Term("a"), Term("b"))),
        ("NOT a", Not(Term("a"))),
        ('"event loop"', Phrase(("event", "loop"))),
        ("a b", And(Term("a"), Term("b"))),
        ("a OR b AND c", Or(Term("a"), And(Term("b"), Term("c")))),
        (
            'python AND (async OR await) NOT java "event loop"',
            Term("python") & (Term("async") | Term("await")) & ~Term("java") & Phrase(("event", "loop")),
        ),
    ],
)
def test_parser_ast(query: str, expected: object) -> None:
    assert parse_query(query) == expected


@pytest.mark.parametrize("bad_query", ["", "   ", "(python", "python)", "AND python"])
def test_parser_invalid_raises(bad_query: str) -> None:
    with pytest.raises(ValueError):
        parse_query(bad_query)


@pytest.mark.parametrize("scorer", [BM25(), TfIdf()])
def test_scorers_rare_vs_common(scorer: Scorer, sample_index: Index) -> None:
    rare = scorer.score("concurrency", Posting(0, 1), sample_index)
    common = scorer.score("python", Posting(0, 1), sample_index)
    assert rare > common


def test_bm25_saturation(sample_index: Index) -> None:
    bm25 = BM25()
    s19 = bm25.score("python", Posting(0, 19), sample_index)
    s20 = bm25.score("python", Posting(0, 20), sample_index)
    assert (s20 - s19) < 0.02


def test_bm25_short_vs_long(sample_index: Index) -> None:
    bm25 = BM25()
    short_score = bm25.score("python", Posting(3, 1), sample_index)
    long_score = bm25.score("python", Posting(4, 1), sample_index)
    assert short_score > long_score


def test_save_load_roundtrip(sample_index: Index, tmp_path: Path) -> None:
    out = tmp_path / "idx.bin"
    save_index(sample_index, out)
    loaded = load_index(out)
    assert loaded == sample_index


def test_open_index_closes_on_error(sample_index: Index, tmp_path: Path) -> None:
    out = tmp_path / "idx.bin"
    save_index(sample_index, out)
    with pytest.raises(RuntimeError), open_index(out) as idx:
        assert idx.num_docs == 5
        raise RuntimeError("fail")
    assert len(idx) == 0


def test_cli_index_search_stats(sample_corpus: Path, tmp_path: Path) -> None:
    idx_file = tmp_path / "index.bin"

    res_idx = runner.invoke(app, ["index", str(sample_corpus), "--out", str(idx_file), "--positions"])
    assert res_idx.exit_code == 0
    assert idx_file.is_file()

    res_stats = runner.invoke(app, ["stats", str(idx_file)])
    assert res_stats.exit_code == 0
    assert "Documents" in res_stats.stdout

    res_json = runner.invoke(app, ["search", str(idx_file), '"event loop"', "--json"])
    assert res_json.exit_code == 0
    data = json.loads(res_json.stdout)
    assert len(data) == 2


def test_cli_missing_index_error(tmp_path: Path) -> None:
    res = runner.invoke(app, ["search", str(tmp_path / "missing.bin"), "python"])
    assert res.exit_code != 0


def test_cli_invalid_query_error(sample_index: Index, tmp_path: Path) -> None:
    idx_file = tmp_path / "index.bin"
    save_index(sample_index, idx_file)
    res = runner.invoke(app, ["search", str(idx_file), "(python"])
    assert res.exit_code != 0


sorted_ids = st.lists(st.integers(min_value=0, max_value=200), unique=True).map(sorted)


@given(a=sorted_ids, b=sorted_ids)
def test_prop_merge_matches_set(a: list[int], b: list[int]) -> None:
    assert merge_and(a, b) == sorted(set(a) & set(b))
    assert merge_or(a, b) == sorted(set(a) | set(b))
    assert merge_not(a, b) == sorted(set(a) - set(b))


@given(
    docs_text=st.lists(
        st.text(alphabet="abcde ", min_size=1, max_size=40),
        min_size=1,
        max_size=10,
    )
)
def test_prop_postings_always_sorted(docs_text: list[str]) -> None:
    postings: dict[str, list[Posting]] = {}
    docs: dict[int, DocMeta] = {}
    for doc_id, text in enumerate(docs_text):
        toks = list(tokenizer(text))
        docs[doc_id] = DocMeta(doc_id, f"{doc_id}.txt", f"d{doc_id}", len(toks))
        freqs: dict[str, int] = {}
        for t in toks:
            freqs[t] = freqs.get(t, 0) + 1
        for t, f in freqs.items():
            postings.setdefault(t, []).append(Posting(doc_id, f))

    idx = Index(postings, docs)
    for term in idx:
        ids = [p.doc_id for p in idx[term]]
        assert ids == sorted(ids)


@given(
    terms=st.dictionaries(
        keys=st.text(alphabet="abc", min_size=1, max_size=5),
        values=st.lists(st.integers(min_value=0, max_value=20), unique=True, min_size=1).map(sorted),
        max_size=10,
    )
)
def test_prop_save_load_identity(tmp_path_factory: pytest.TempPathFactory, terms: dict[str, list[int]]) -> None:
    postings = {t: [Posting(d, 1) for d in ids] for t, ids in terms.items()}
    all_ids = {d for ids in terms.values() for d in ids}
    docs = {d: DocMeta(d, f"{d}.txt", f"d{d}", 1) for d in all_ids}
    idx = Index(postings, docs)

    path = tmp_path_factory.mktemp("hyp") / "idx.bin"
    save_index(idx, path)
    assert load_index(path) == idx