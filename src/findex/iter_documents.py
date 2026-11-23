from collections.abc import Iterator
from pathlib import Path


def read_file(input_path: str | Path) -> Iterator[tuple[str, str]]:
    files_list = sorted(Path(input_path).glob("*.txt"))
    for file in files_list:
        with open(file, "r", encoding="utf-8") as f:
            data = f.read()
        yield file.name, data