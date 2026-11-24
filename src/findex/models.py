from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    freq: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    path: str
    title: str
    length: int


@dataclass(order=True, frozen=True, slots=True)
class SearchResult:
    score: float
    doc_id: int
    title: str
    path: str
    snippet: str = ""


class Index(Mapping[str, list[Posting]]):
    def __init__(
        self,
        postings: dict[str, list[Posting]],
        docs: dict[int, DocMeta],
        has_positions: bool = False,
    ) -> None:
        self._postings = postings
        self.docs = docs
        self.has_positions = has_positions
        self._closed = False
        self._doc_lengths_np: np.ndarray | None = None
        self._postings_np: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def __getitem__(self, term: str) -> list[Posting]:
        return self._postings[term]

    def __iter__(self) -> Iterator[str]:
        return iter(self._postings)

    def __len__(self) -> int:
        return len(self._postings)

    def __contains__(self, term: object) -> bool:
        return term in self._postings

    def __repr__(self) -> str:
        return (
            f"Index(terms={len(self._postings)}, "
            f"docs={self.num_docs}, "
            f"has_positions={self.has_positions})"
        )
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Index):
            return NotImplemented
        return (
            self._postings == other._postings
            and self.docs == other.docs
            and self.has_positions == other.has_positions
        )

    @property
    def num_docs(self) -> int:
        return len(self.docs)

    @cached_property
    def avg_doc_length(self) -> float:
        if not self.docs:
            return 0.0
        return sum(d.length for d in self.docs.values()) / len(self.docs)

    def df(self, term: str) -> int:
        postings = self._postings.get(term)
        return len(postings) if postings else 0

    def close(self) -> None:
        if not self._closed:
            self._postings.clear()
            self.docs.clear()
            self.__dict__.pop("avg_doc_length", None)
            self._closed = True
    @property
    def doc_lengths_np(self) -> NDArray[np.int32]:
        arr: NDArray[np.int32] | None = getattr(self, "_doc_lengths_np", None)
        if arr is None:
            size = (max(self.docs.keys()) + 1) if self.docs else 0
            arr = np.zeros(size, dtype=np.int32)
            for doc_id, meta in self.docs.items():
                arr[doc_id] = meta.length
            self._doc_lengths_np = arr
        return arr

    def posting_arrays(self, term: str) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        cache: dict[str, tuple[NDArray[np.int32], NDArray[np.int32]]] | None = getattr(
            self, "_postings_np", None
        )
        if cache is None:
            cache = {}
            self._postings_np = cache
        cached = cache.get(term)
        if cached is not None:
            return cached
        plist = self._postings.get(term, ())
        if not plist:
            empty: NDArray[np.int32] = np.empty(0, dtype=np.int32)
            pair = (empty, empty)
        else:
            doc_ids = np.fromiter((p.doc_id for p in plist), dtype=np.int32, count=len(plist))
            freqs = np.fromiter((p.freq for p in plist), dtype=np.int32, count=len(plist))
            pair = (doc_ids, freqs)
        cache[term] = pair
        return pair