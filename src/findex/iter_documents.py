import json
from collections.abc import Iterator
from pathlib import Path


def read_file(input_path: str | Path) -> Iterator[tuple[str, str]]:
    p = Path(input_path)
    if p.is_file() and p.suffix == ".jsonl":
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                yield str(rec.get("title") or rec["url"]), str(rec.get("text", ""))
        return

    files_list = sorted(p.glob("*.txt"))
    for file in files_list:
        with open(file, "r", encoding="utf-8") as f:
            data = f.read()
        yield file.name, data