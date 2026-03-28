import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.transform import transform_step


def transform_all_processed_data():
    processed_dir = Path("data/processed")
    transformed_dir = Path("data/transformed")
    transformed_dir.mkdir(parents=True, exist_ok=True)

    processed_files = sorted(processed_dir.glob("episode_*.json"))

    if not processed_files:
        print("No processed files found.")
        return

    all_samples = []

    for file in processed_files:
        with open(file, "r", encoding="utf-8") as f:
            episode = json.load(f)

        for step in episode:
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