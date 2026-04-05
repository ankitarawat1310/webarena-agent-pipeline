import json
import random
from pathlib import Path


def split_dataset():
    input_file = Path("data/processed/sft_data.jsonl")
    output_dir = Path("data/splits")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        print(f"Dataset file not found: {input_file}")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    print(f"Total samples: {len(samples)}")

    random.shuffle(samples)

    total = len(samples)
    train_end = int(0.7 * total)
    val_end = int(0.85 * total)

    train_data = samples[:train_end]
    val_data = samples[train_end:val_end]
    test_data = samples[val_end:]

    def save_split(data, filename):
        with open(output_dir / filename, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    save_split(train_data, "train.jsonl")
    save_split(val_data, "val.jsonl")
    save_split(test_data, "test.jsonl")

    print("Split complete:")
    print(f"Train: {len(train_data)}")
    print(f"Validation: {len(val_data)}")
    print(f"Test: {len(test_data)}")


if __name__ == "__main__":
    split_dataset()