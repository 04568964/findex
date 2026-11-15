from pathlib import Path

def read_file(input_path):
    files_list = sorted(Path(input_path).glob("*.txt"))
    for file in files_list:
        with open(file, "r", encoding="utf-8") as f:
            data = f.read()
        yield file.name, data

def greedy_read_files(input_path):
    files_list = Path(input_path).glob("*.txt")
    all_texts = []
    for file in files_list:
        with open(file, "r", encoding="utf-8") as f:
            all_texts.append(f.read())
    return all_texts