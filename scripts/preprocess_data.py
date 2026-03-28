import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import is_valid_step, preprocess_step


def preprocess_all_raw_data():
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(raw_dir.glob("episode_*.json"))

    if not raw_files:
        print("No raw episode files found.")
        return

    for raw_file in raw_files:
        with open(raw_file, "r", encoding="utf-8") as f:
            episode = json.load(f)

        cleaned_episode = []

        for step in episode:
            if is_valid_step(step):
                cleaned_episode.append(preprocess_step(step))
            else:
                print(f"Skipping invalid step in {raw_file.name}")

        output_file = processed_dir / raw_file.name
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(cleaned_episode, f, indent=2)

        print(f"Processed: {raw_file.name} -> {output_file}")


if __name__ == "__main__":
    preprocess_all_raw_data()