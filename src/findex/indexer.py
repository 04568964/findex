from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from findex.models import DocMeta, Index, Posting
from findex.timing import timed
from findex.tokenizer import tokenizer

ExecutorName = Literal["serial", "threads", "processes"]


def build_partial(
    paths: list[Path],
    start_doc_id: int,
    with_positions: bool = False,
) -> Index:
    postings_map: dict[str, list[Posting]] = defaultdict(list)
    docs: dict[int, DocMeta] = {}

    for offset, full_path in enumerate(paths):
        doc_id = start_doc_id + offset
        text = full_path.read_text(encoding="utf-8")
        tokens = list(tokenizer(text))
        docs[doc_id] = DocMeta(
            doc_id=doc_id,
            path=str(full_path),
            title=full_path.stem,
            length=len(tokens),
        )

        if with_positions:
            pos_map: dict[str, list[int]] = defaultdict(list)
            for pos, tok in enumerate(tokens):
                pos_map[tok].append(pos)
            for term, positions in pos_map.items():
                postings_map[term].append(
                    Posting(doc_id=doc_id, freq=len(positions), positions=tuple(positions))
                )
        else:
            freq_map: dict[str, int] = defaultdict(int)
            for tok in tokens:
                freq_map[tok] += 1
            for term, freq in freq_map.items():
                postings_map[term].append(Posting(doc_id=doc_id, freq=freq))

    return Index(dict(postings_map), docs, has_positions=with_positions)


@timed
def merge_indexes(partials: list[Index]) -> Index:
    if not partials:
        return Index({}, {}, has_positions=False)

    has_positions = partials[0].has_positions
    ordered = sorted(partials, key=lambda idx: min(idx.docs) if idx.docs else -1)

    merged_postings: dict[str, list[Posting]] = defaultdict(list)
    merged_docs: dict[int, DocMeta] = {}

    for part in ordered:
        merged_docs.update(part.docs)
        for term, plist in part.items():
            merged_postings[term].extend(plist)

    return Index(dict(merged_postings), merged_docs, has_positions=has_positions)


def _split_chunks(paths: list[Path], workers: int) -> list[tuple[int, list[Path]]]:
    n = len(paths)
    if n == 0:
        return []
    k = max(1, min(workers, n))
    base, rem = divmod(n, k)
    chunks: list[tuple[int, list[Path]]] = []
    start = 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        chunks.append((start, paths[start : start + size]))
        start += size
    return chunks


@timed
def build_index(
    data_dir: str | Path,
    with_positions: bool = False,
    workers: int = 1,
    executor: ExecutorName = "serial",
) -> Index:
    all_paths = sorted(Path(data_dir).glob("*.txt"))
    if not all_paths:
        return Index({}, {}, has_positions=with_positions)

    chunks = _split_chunks(all_paths, workers)

    if executor == "serial" or len(chunks) == 1:
        partials = [build_partial(chunk, start_id, with_positions) for start_id, chunk in chunks]
        return merge_indexes(partials)

    pool_cls = ThreadPoolExecutor if executor == "threads" else ProcessPoolExecutor
    with pool_cls(max_workers=len(chunks)) as pool:
        futures = [
            pool.submit(build_partial, chunk, start_id, with_positions)
            for start_id, chunk in chunks
        ]
        partials = [f.result() for f in futures]

    return merge_indexes(partials)