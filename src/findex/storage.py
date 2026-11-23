import pickle
import json
from dataclasses import asdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from findex.models import Posting, DocMeta
from findex.models import Index
from findex.timing import timed
def save_index_pickle(inv_index, doc_meta, filepath):
    with open(filepath, "wb") as f:
        pickle.dump({"index" : inv_index, "meta" : doc_meta}, f)

def load_index_pickle(filepath):
    '''
    load небезпечний бо під час завантаження файлу pickle дозволяє
    виконання коду через __reduce__
    '''
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data["index"], data["meta"]

def save_index_json(inv_index, doc_meta, filepath):
    serializable_index = {
        term: [asdict(p) for p in postings] 
        for term, postings in inv_index.items()
    }
    serializable_meta = {
        doc_id: asdict(meta) 
        for doc_id, meta in doc_meta.items()
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({"index": serializable_index, "meta": serializable_meta}, f)

def load_index_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    inv_index = {
        term: [Posting(**p) for p in postings] 
        for term, postings in data["index"].items()
    }
    doc_meta = {
        int(doc_id): DocMeta(**meta) 
        for doc_id, meta in data["meta"].items()
    }
    return inv_index, doc_meta


@timed
def load_index(path: str | Path) -> Index:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, Index):
        raise TypeError(f"Expected Index, got {type(obj).__name__}")
    return obj

@contextmanager
def open_index(path: str | Path) -> Iterator[Index]:
    index = load_index(path)
    try:
        yield index
    finally:
        index.close()

@timed
def save_index(index: Index, path: str | Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)