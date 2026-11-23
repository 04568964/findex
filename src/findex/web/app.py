import logging
import math
import re
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, cast

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape

from findex.models import Index
from findex.scorers import BM25, Scorer, TfIdf
from findex.searcher import Searcher
from findex.storage import open_index
from findex.tokenizer import tokenizer
from findex.web.config import Settings
from findex.web.schemas import (
    DocResponse,
    HealthResponse,
    ScorerName,
    SearchHit,
    SearchResponse,
    StatsResponse,
)

logger = logging.getLogger("findex.web")
TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _highlight(snippet: str, query: str) -> Markup:
    terms = sorted(set(tokenizer(query)), key=len, reverse=True)
    safe = str(escape(snippet))
    if not terms:
        return Markup(safe)
    pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b", re.IGNORECASE)
    return Markup(pattern.sub(r"<mark>\1</mark>", safe))


cast(dict[str, object], templates.env.filters)["highlight"] = _highlight


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    settings = Settings()
    app.state.index = None
    if settings.index_path.is_file():
        with open_index(settings.index_path) as idx:
            app.state.index = idx
            logger.info("Loaded index from %s (%d docs)", settings.index_path, idx.num_docs)
            yield
        app.state.index = None
    else:
        logger.warning("Index file not found: %s", settings.index_path)
        yield


app = FastAPI(title="findex", version="0.7.0", lifespan=lifespan)


@app.middleware("http")
async def request_id_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request.state.request_id = req_id
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "Unhandled error [request_id=%s] %s %s", req_id, request.method, request.url.path
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": req_id},
            headers={"X-Request-ID": req_id},
        )
    response.headers["X-Request-ID"] = req_id
    return response


def get_index_optional(request: Request) -> Index | None:
    return getattr(request.app.state, "index", None)


def get_index(idx: Annotated[Index | None, Depends(get_index_optional)]) -> Index:
    if idx is None:
        raise HTTPException(status_code=503, detail="Index is not loaded")
    return idx


def _make_scorer(name: ScorerName) -> Scorer:
    return BM25() if name == "bm25" else TfIdf()


def _execute_search(idx: Index, q: str, k: int, scorer: ScorerName, page: int) -> SearchResponse:
    searcher = Searcher(idx, scorer=_make_scorer(scorer))
    try:
        all_results = searcher.search(q, top_k=idx.num_docs or 1, with_snippets=True)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    total = len(all_results)
    total_pages = max(1, math.ceil(total / k))
    start = (page - 1) * k
    page_items = all_results[start : start + k]

    hits = [
        SearchHit(
            doc_id=r.doc_id,
            score=round(r.score, 6),
            title=r.title,
            path=r.path,
            snippet=r.snippet,
        )
        for r in page_items
    ]
    return SearchResponse(
        query=q,
        scorer=scorer,
        page=page,
        page_size=k,
        total=total,
        total_pages=total_pages,
        results=hits,
    )


@app.get("/health", response_model=HealthResponse)
def health(idx: Annotated[Index | None, Depends(get_index_optional)]) -> JSONResponse | HealthResponse:
    if idx is None:
        return JSONResponse(
            status_code=503,
            content=HealthResponse(status="unavailable", index_loaded=False).model_dump(),
        )
    return HealthResponse(status="ok", index_loaded=True)


@app.get("/stats", response_model=StatsResponse)
def stats(idx: Annotated[Index, Depends(get_index)]) -> StatsResponse:
    return StatsResponse(
        documents=idx.num_docs,
        unique_terms=len(idx),
        avg_doc_length=round(idx.avg_doc_length, 2),
        has_positions=idx.has_positions,
    )


@app.get("/search", response_model=SearchResponse)
def search_api(
    idx: Annotated[Index, Depends(get_index)],
    q: Annotated[str, Query(min_length=1, max_length=500)],
    k: Annotated[int, Query(ge=1, le=100)] = 10,
    scorer: Annotated[ScorerName, Query()] = "bm25",
    page: Annotated[int, Query(ge=1)] = 1,
) -> SearchResponse:
    return _execute_search(idx, q, k, scorer, page)


@app.get("/docs/{doc_id}", response_model=DocResponse)
def get_doc_api(
    doc_id: int,
    idx: Annotated[Index, Depends(get_index)],
) -> DocResponse:
    meta = idx.docs.get(doc_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    text = ""
    p = Path(meta.path)
    if p.is_file():
        text = p.read_text(encoding="utf-8", errors="replace")

    return DocResponse(
        doc_id=meta.doc_id,
        title=meta.title,
        path=meta.path,
        length=meta.length,
        text=text,
    )


@app.get("/", response_class=HTMLResponse)
def home_page(
    request: Request,
    idx: Annotated[Index | None, Depends(get_index_optional)],
    q: Annotated[str | None, Query()] = None,
    k: Annotated[int, Query(ge=1, le=50)] = 10,
    scorer: Annotated[ScorerName, Query()] = "bm25",
    page: Annotated[int, Query(ge=1)] = 1,
) -> HTMLResponse:
    search_res: SearchResponse | None = None
    error: str | None = None

    if q and q.strip():
        if idx is None:
            error = "Index is not loaded."
        else:
            try:
                search_res = _execute_search(idx, q.strip(), k, scorer, page)
            except HTTPException as exc:
                error = str(exc.detail)

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "q": q or "",
            "k": k,
            "scorer": scorer,
            "page": page,
            "res": search_res,
            "error": error,
            "num_docs": idx.num_docs if idx else 0,
        },
    )


@app.get("/view/{doc_id}", response_class=HTMLResponse)
def view_doc_page(
    request: Request,
    doc_id: int,
    idx: Annotated[Index, Depends(get_index)],
) -> HTMLResponse:
    doc = get_doc_api(doc_id, idx)
    return templates.TemplateResponse(request, "doc.html", {"doc": doc})