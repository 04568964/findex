from dataclasses import dataclass

@dataclass(frozen=True,slots=True)
class Posting:
    doc_id: int
    frequensy: int

@dataclass(frozen=True,slots=True)
class DocMeta:
    doc_id: int
    filename: str
    length: int 
