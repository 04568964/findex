from pathlib import Path

import pytest

from findex.indexer import build_index
from findex.models import Index


@pytest.fixture
def sample_corpus(tmp_path: Path) -> Path:
    docs = {
        "doc1.txt": "Python async await event loop concurrency",
        "doc2.txt": "Java virtual machine garbage collector",
        "doc3.txt": "Python university research science event loop",
        "doc4.txt": "Short python note",
        "doc5.txt": " ".join(["filler"] * 80 + ["python"]),
    }
    for name, content in docs.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    return tmp_path


@pytest.fixture
def sample_index(sample_corpus: Path) -> Index:
    return build_index(sample_corpus, with_positions=True)