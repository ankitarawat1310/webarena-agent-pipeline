import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.transform import transform_step


def transform_all_processed_data():
    input_file = Path("data/processed/processed_data.jsonl")
    transformed_dir = Path("data/transformed")
    transformed_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        return

    all_samples = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            step = json.loads(line)
            sample = transform_step(step)
            all_samples.append(sample)

    output_file = transformed_dir / "dataset.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Saved transformed dataset: {output_file}")
    print(f"Total samples: {len(all_samples)}")


if __name__ == "__main__":
    transform_all_processed_data()