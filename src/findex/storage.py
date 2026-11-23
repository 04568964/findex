import pickle
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from findex.models import Index
from findex.timing import timed


@timed
def save_index(index: Index, path: str | Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


@timed
def load_index(path: str | Path) -> Index:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, Index):
        raise TypeError(f"Expected Index, got {type(obj).__name__}")
    return obj


@contextmanager
def open_index(path: str | Path) -> Generator[Index]:
    index = load_index(path)
    try:
        yield index
    finally:
        index.close()