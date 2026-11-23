from pathlib import Path

import pytest

from findex.indexer import ExecutorName, build_index, build_partial, merge_indexes


def test_merge_single_partial_matches_serial(sample_corpus: Path) -> None:
    all_paths = sorted(sample_corpus.glob("*.txt"))
    expected = build_index(sample_corpus, with_positions=True, workers=1, executor="serial")
    actual = merge_indexes([build_partial(all_paths, start_doc_id=0, with_positions=True)])
    assert actual == expected


@pytest.mark.parametrize("executor", ["serial", "threads", "processes"])
@pytest.mark.parametrize("workers", [1, 2, 4])
def test_parallel_executors_match_serial(
    sample_corpus: Path,
    executor: ExecutorName,
    workers: int,
) -> None:
    expected = build_index(sample_corpus, with_positions=True, workers=1, executor="serial")
    actual = build_index(sample_corpus, with_positions=True, workers=workers, executor=executor)
    assert actual == expected


@pytest.mark.parametrize("executor", ["threads", "processes"])
def test_worker_failure_propagates(
    tmp_path: Path,
    executor: ExecutorName,
) -> None:
    (tmp_path / "good.txt").write_text("valid text", encoding="utf-8")
    (tmp_path / "bad.txt").write_bytes(b"\xff\xfe\xfd")

    with pytest.raises(UnicodeDecodeError):
        build_index(tmp_path, workers=2, executor=executor)