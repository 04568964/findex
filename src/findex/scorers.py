import math
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from findex.models import Index, Posting


class Scorer(Protocol):
    def score(self, term: str, posting: Posting, index: Index) -> float: ...
    def score_array(
    self,
    doc_ids: NDArray[np.int32],
    freqs: NDArray[np.int32],
    doc_lengths: NDArray[np.int32],
    num_docs: int,
    avg_dl: float,
) -> NDArray[np.float64]: ...


class TfIdf:
    def score(self, term: str, posting: Posting, index: Index) -> float:
        df = index.df(term)
        if df == 0 or posting.freq == 0:
            return 0.0
        tf = 1.0 + math.log(posting.freq)
        idf = math.log((index.num_docs + 1) / (df + 1)) + 1.0
        return tf * idf
    def score_array(
        self,
        doc_ids: NDArray[np.int32],
        freqs: NDArray[np.int32],
        doc_lengths: NDArray[np.int32],
        num_docs: int,
        avg_dl: float,
    ) -> NDArray[np.float64]:
        df = len(doc_ids)
        if df == 0 or num_docs == 0:
            return np.empty(0, dtype=np.float64)
        idf = math.log((num_docs + 1.0) / (df + 1.0)) + 1.0
        return (1.0 + np.log(freqs.astype(np.float64))) * idf


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def score(self, term: str, posting: Posting, index: Index) -> float:
        df = index.df(term)
        if df == 0 or posting.freq == 0:
            return 0.0
        n = index.num_docs
        idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
        dl = index.docs[posting.doc_id].length
        avgdl = index.avg_doc_length or 1.0
        tf = posting.freq
        denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / avgdl))
        return idf * (tf * (self.k1 + 1.0)) / denom
    def score_array(
        self,
        doc_ids: NDArray[np.int32],
        freqs: NDArray[np.int32],
        doc_lengths: NDArray[np.int32],
        num_docs: int,
        avg_dl: float,
    ) -> NDArray[np.float64]:
        df = len(doc_ids)
        if df == 0 or num_docs == 0:
            return np.empty(0, dtype=np.float64)
        idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
        tf = freqs.astype(np.float64)
        dl = doc_lengths[doc_ids].astype(np.float64)
        denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / avg_dl))
        return idf * (tf * (self.k1 + 1.0)) / denom