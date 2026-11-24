from typing import Literal

from pydantic import BaseModel

ScorerName = Literal["bm25", "tfidf"]
SearchMode = Literal["lexical", "semantic", "hybrid"]


class SearchHit(BaseModel):
    doc_id: int
    score: float
    title: str
    path: str
    snippet: str


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode = "lexical"
    scorer: ScorerName
    page: int
    page_size: int
    total: int
    total_pages: int
    results: list[SearchHit]


class DocResponse(BaseModel):
    doc_id: int
    title: str
    path: str
    length: int
    text: str


class StatsResponse(BaseModel):
    documents: int
    unique_terms: int
    avg_doc_length: float
    has_positions: bool


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    index_loaded: bool