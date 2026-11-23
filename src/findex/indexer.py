from collections import defaultdict
from pathlib import Path

from findex.iter_documents import read_file
from findex.models import DocMeta, Index, Posting
from findex.timing import timed
from findex.tokenizer import tokenizer


@timed
def build_index(data_dir: str | Path, with_positions: bool = False) -> Index:
    postings_map: dict[str, list[Posting]] = defaultdict(list)
    docs: dict[int, DocMeta] = {}
    base_dir = Path(data_dir)

    for doc_id, (filename, text) in enumerate(read_file(base_dir)):
        full_path = base_dir / filename
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
