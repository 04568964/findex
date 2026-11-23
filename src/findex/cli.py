import json
import logging
import sys
from pathlib import Path
from typing import Annotated, Literal, NoReturn

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

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
) -> None:
    if not index_file.is_file():
        _fail(f"Index file not found: {index_file}")

    scorer_obj: Scorer = BM25() if scorer == "bm25" else TfIdf()

    try:
        with open_index(index_file) as idx:
            searcher = Searcher(idx, scorer=scorer_obj)
            results = searcher.search(query, top_k=top_k, with_snippets=not no_snippets)
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

    table = Table(title=f"Results for '{query}' ({scorer})")
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