from datasets import load_dataset
from pathlib import Path

DATASET_PATH = Path(r"data")

def main():
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True, trust_remote_code=True)
    dataset = dataset.shuffle(buffer_size=10000, seed=42)
    wiki_iter = iter(dataset)

    DATASET_PATH.mkdir(parents=True, exist_ok=True)

    idx = 0

    while idx <= 1000:
        article = next(wiki_iter)

        if len(article['text']) < 1000:
            continue

        filename = f"doc_{idx:04d}.txt"
        file_path = DATASET_PATH.joinpath(filename)
        with open(file_path,"w", encoding = "utf-8") as f:
            f.write(article['text'])
        idx +=1

if __name__ == "__main__":
    main()