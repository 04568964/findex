import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Annotated, Literal, NoReturn

import typer
from rich.console import Console
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from findex.crawler import CrawlStats, crawl
from findex.indexer import ExecutorName, build_index
from findex.scorers import BM25, Scorer, TfIdf
from findex.searcher import Searcher
from findex.storage import open_index, save_index

ScorerName = Literal["bm25", "tfidf"]

app = typer.Typer(add_completion=False, help="Full-text search engine CLI.")
err_console = Console(stderr=True)
out_console = Console()


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format="%(levelname)s:%(name)s:%(message)s",
        force=True,
    )


def _fail(message: str, code: int = 1) -> NoReturn:
    err_console.print(f"Error: {message}")
    raise typer.Exit(code=code)


@app.callback()
def main(
    verbose: Annotated[
        int,
        typer.Option("-v", "--verbose", count=True, help="Increase log verbosity (-v, -vv)."),
    ] = 0,
) -> None:
    _configure_logging(verbose)


@app.command()
def index(
    data_dir: Annotated[Path, typer.Argument(help="Directory with .txt documents.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output index file.")] = Path("index.bin"),
    positions: Annotated[bool, typer.Option("--positions", help="Store token positions.")] = False,
    workers: Annotated[
        int,
        typer.Option("--workers", "-w", min=1, help="Number of worker threads/processes."),
    ] = 1,
    executor: Annotated[
        ExecutorName,
        typer.Option("--executor", "-e", help="Execution strategy."),
    ] = "serial",
) -> None:
    if not data_dir.is_dir():
        _fail(f"Data directory not found: {data_dir}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        task = progress.add_task(f"Building index ({executor}, workers={workers})...", total=None)
        idx = build_index(data_dir, with_positions=positions, workers=workers, executor=executor)
        progress.update(task, description="Saving index...")
        save_index(idx, out)

    err_console.print(f"Saved {len(idx)} terms ({idx.num_docs} docs) to {out}")


@app.command()
def search(
    index_file: Annotated[Path, typer.Argument(help="Path to index file.")],
    query: Annotated[str, typer.Argument(help="Search query.")],
    scorer: Annotated[ScorerName, typer.Option("--scorer", help="Scoring algorithm.")] = "bm25",
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="Number of results.")] = 5,
    as_json: Annotated[bool, typer.Option("--json", help="Output results as JSON to stdout.")] = False,
    no_snippets: Annotated[bool, typer.Option("--no-snippets", help="Disable snippets.")] = False,
    mode: Annotated[str, typer.Option("--mode", help="lexical | semantic | hybrid")] = "lexical",
    embeddings: Annotated[Path | None, typer.Option("--embeddings", help="Path to embeddings .npy")] = None
) -> None:
    if not index_file.is_file():
        _fail(f"Index file not found: {index_file}")

    scorer_obj: Scorer = BM25() if scorer == "bm25" else TfIdf()

    try:
        with open_index(index_file) as idx:
            if mode == "lexical":
                searcher = Searcher(idx, scorer=scorer_obj)
                results = searcher.search(query, top_k=top_k, with_snippets=not no_snippets)
            else:
                emb_path = embeddings or Path("deploy/embeddings.npy")
                if not emb_path.is_file():
                    _fail(f"Embeddings file not found: {emb_path}")

                from findex.models import SearchResult
                from findex.semantic import FastEmbedder, SemanticIndex, reciprocal_rank_fusion

                sem_idx = SemanticIndex.load(emb_path)
                embedder = FastEmbedder()
                q_vec = embedder.embed_query(query)

                if mode == "semantic":
                    ranked = sem_idx.search(q_vec, top_k=top_k)
                else:
                    searcher = Searcher(idx, scorer=scorer_obj)
                    lex_hits = searcher.search(query, top_k=idx.num_docs or 1, with_snippets=False)
                    sem_hits = sem_idx.search(q_vec, top_k=idx.num_docs or 1)
                    ranked = reciprocal_rank_fusion(
                        [r.doc_id for r in lex_hits],
                        [doc_id for doc_id, _ in sem_hits],
                        top_k=top_k,
                    )

                results = [
                    SearchResult(
                        score=score,
                        doc_id=doc_id,
                        title=idx.docs[doc_id].title,
                        path=idx.docs[doc_id].path,
                        snippet="",
                    )
                    for doc_id, score in ranked
                    if doc_id in idx.docs
                ]
    except (ValueError, RuntimeError) as exc:
        _fail(str(exc))

    if as_json:
        payload = [
            {
                "doc_id": r.doc_id,
                "score": round(r.score, 6),
                "title": r.title,
                "path": r.path,
                "snippet": r.snippet,
            }
            for r in results
        ]
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return

    table = Table(title=f"Results for '{query}' ({mode}, {scorer})")
    table.add_column("Score", justify="right")
    table.add_column("Title")
    table.add_column("Snippet")
    for r in results:
        table.add_row(f"{r.score:.4f}", r.title, r.snippet)
    out_console.print(table)


@app.command()
def stats(
    index_file: Annotated[Path, typer.Argument(help="Path to index file.")],
) -> None:
    if not index_file.is_file():
        _fail(f"Index file not found: {index_file}")

    with open_index(index_file) as idx:
        table = Table(title=f"Index stats: {index_file}")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_row("Documents", str(idx.num_docs))
        table.add_row("Unique terms", str(len(idx)))
        table.add_row("Avg doc length", f"{idx.avg_doc_length:.2f}")
        table.add_row("Has positions", str(idx.has_positions))
        out_console.print(table)

@app.command(name="crawl")
def crawl_cmd(
    start_url: Annotated[str, typer.Argument(help="Starting URL to crawl.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output JSONL file.")] = Path("pages.jsonl"),
    max_pages: Annotated[int, typer.Option("--max-pages", min=1, help="Maximum pages to crawl.")] = 200,
    concurrency: Annotated[int, typer.Option("--concurrency", "-c", min=1, help="Global concurrency.")] = 5,
    per_host: Annotated[int, typer.Option("--per-host", min=1, help="Max requests per host.")] = 2,
    delay: Annotated[float, typer.Option("--delay", min=0.0, help="Min delay per host (s).")] = 0.2,
    log: Annotated[Path, typer.Option("--log", help="Crawl log file.")] = Path("crawl.log"),
    debug: Annotated[bool, typer.Option("--debug", help="Run asyncio event loop in debug mode.")] = False,
) -> None:
    stats = CrawlStats()

    def _render_table() -> Table:
        table = Table(title="findex crawl")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_row("Pages collected", f"{stats.fetched} / {max_pages}")
        table.add_row("Pages / sec", f"{stats.pages_per_sec:.2f}")
        table.add_row("In-flight", str(stats.in_flight))
        table.add_row("Errors", str(stats.errors))
        table.add_row("Queue depth", str(stats.queue_size))
        return table

    async def _run() -> None:
        with Live(_render_table(), console=err_console, refresh_per_second=4) as live:
            async for page in crawl(
                start_url,
                max_pages=max_pages,
                concurrency=concurrency,
                per_host_concurrency=per_host,
                host_delay=delay,
                stats=stats,
                log_path=log,
            ):
                record = {
                    "url": page.url,
                    "title": page.title,
                    "text": page.text,
                    "status": page.status,
                    "size_bytes": page.size_bytes,
                    "elapsed_s": round(page.elapsed_s, 4),
                }
                line = json.dumps(record, ensure_ascii=False) + "\n"
                await asyncio.to_thread(f.write, line)
                await asyncio.to_thread(f.flush)
                live.update(_render_table())
            live.update(_render_table())

    with open(out, "w", encoding="utf-8") as f:
        asyncio.run(_run(), debug=debug)

    err_console.print(f"Crawled {stats.fetched} pages ({stats.errors} errors) -> {out}")

@app.command()
def serve(
    host: Annotated[str, typer.Option("--host", help="Bind host.")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", "-p", min=1, max=65535, help="Bind port.")] = 8000,
    workers: Annotated[int, typer.Option("--workers", "-w", min=1, help="Uvicorn worker processes.")] = 1,
    index_path: Annotated[Path | None, typer.Option("--index", help="Path to index file (overrides INDEX_PATH).")] = None,
) -> None:
    import os

    import uvicorn

    if index_path is not None:
        os.environ["INDEX_PATH"] = str(index_path)

    uvicorn.run("findex.web.app:app", host=host, port=port, workers=workers)

@app.command()
def embed(
    index_path: Annotated[Path, typer.Argument(help="Path to index.bin")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output .npy path")] = Path(
        "deploy/embeddings.npy"
    ),
    chunk_words: Annotated[int, typer.Option("--chunk-words", min=20)] = 120,
    overlap: Annotated[int, typer.Option("--overlap", min=0)] = 20,
) -> None:
    if not index_path.is_file():
        _fail(f"Index file not found: {index_path}")
    from findex.semantic import build_semantic_index

    with open_index(index_path) as idx:
        sem_idx = build_semantic_index(idx, chunk_words=chunk_words, overlap=overlap)
        sem_idx.save(out)
    typer.echo(
        f"Saved {sem_idx.vectors.shape[0]} chunk embeddings (dim={sem_idx.vectors.shape[1]}) to {out}"
    )