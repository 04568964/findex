from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from findex.models import Index

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    def embed_query(self, text: str) -> NDArray[np.float32]: ...
    def embed_texts(self, texts: list[str]) -> NDArray[np.float32]: ...


class FastEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def embed_texts(self, texts: list[str]) -> NDArray[np.float32]:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        raw = np.vstack(list(self._model.embed(texts))).astype(np.float32, copy=False)
        return _l2_normalize(raw)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        vec = next(iter(self._model.embed([text]))).astype(np.float32, copy=False)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0.0 else vec


def _l2_normalize(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (matrix / norms).astype(np.float32, copy=False)


def chunk_text(text: str, chunk_words: int = 120, overlap: int = 20) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_words - overlap)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        part = words[start : start + chunk_words]
        chunks.append(" ".join(part))
        if start + chunk_words >= len(words):
            break
    return chunks


@dataclass(slots=True)
class SemanticIndex:
    vectors: NDArray[np.float32]
    chunk_doc_ids: NDArray[np.int32]

    def save(self, vectors_path: Path) -> None:
        vectors_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(vectors_path, self.vectors)
        ids_path = _doc_ids_path(vectors_path)
        np.save(ids_path, self.chunk_doc_ids)

    @classmethod
    def load(cls, vectors_path: Path) -> "SemanticIndex":
        vectors: NDArray[np.float32] = np.load(vectors_path).astype(np.float32, copy=False)
        ids_path = _doc_ids_path(vectors_path)
        doc_ids: NDArray[np.int32] = np.load(ids_path).astype(np.int32, copy=False)
        return cls(vectors=vectors, chunk_doc_ids=doc_ids)

    def search(self, query_vec: NDArray[np.float32], top_k: int = 10) -> list[tuple[int, float]]:
        if self.vectors.size == 0 or top_k <= 0:
            return []
        sims = self.vectors @ query_vec

        best_per_doc: dict[int, float] = {}
        for idx, sim in enumerate(sims):
            doc_id = int(self.chunk_doc_ids[idx])
            score = float(sim)
            prev = best_per_doc.get(doc_id)
            if prev is None or score > prev:
                best_per_doc[doc_id] = score

        ranked = sorted(best_per_doc.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:top_k]


def _doc_ids_path(vectors_path: Path) -> Path:
    return vectors_path.with_name(f"{vectors_path.stem}_doc_ids.npy")


def build_semantic_index(
    index: Index,
    embedder: Embedder | None = None,
    chunk_words: int = 120,
    overlap: int = 20,
) -> SemanticIndex:
    enc: Embedder = embedder or FastEmbedder()
    chunks: list[str] = []
    doc_ids: list[int] = []

    for doc_id, meta in sorted(index.docs.items()):
        p = Path(meta.path)
        text = p.read_text(encoding="utf-8", errors="replace") if p.is_file() else meta.title
        doc_chunks = chunk_text(text, chunk_words=chunk_words, overlap=overlap)
        if not doc_chunks:
            doc_chunks = [meta.title]
        chunks.extend(doc_chunks)
        doc_ids.extend([doc_id] * len(doc_chunks))

    vectors = enc.embed_texts(chunks)
    return SemanticIndex(
        vectors=vectors,
        chunk_doc_ids=np.asarray(doc_ids, dtype=np.int32),
    )


def reciprocal_rank_fusion(
    lexical_doc_ids: list[int],
    semantic_doc_ids: list[int],
    k: int = 60,
    top_k: int = 10,
) -> list[tuple[int, float]]:
    scores: dict[int, float] = defaultdict(float)
    for rank, doc_id in enumerate(lexical_doc_ids, start=1):
        scores[doc_id] += 1.0 / (k + rank)
    for rank, doc_id in enumerate(semantic_doc_ids, start=1):
        scores[doc_id] += 1.0 / (k + rank)

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:top_k]