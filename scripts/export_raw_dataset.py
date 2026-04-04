import json
from pathlib import Path


def export_raw_dataset():
    raw_dir = Path("data/raw")
    output_file = Path("data/raw_dataset.jsonl")

    raw_files = sorted(raw_dir.glob("episode_*.json"))

    if not raw_files:
        print("No raw episode files found.")
        return

    total_records = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        for raw_file in raw_files:
            with open(raw_file, "r", encoding="utf-8") as in_f:
                episode = json.load(in_f)

            for step in episode:
                out_f.write(json.dumps(step) + "\n")
                total_records += 1

    print(f"Created combined dataset: {output_file}")
    print(f"Total records: {total_records}")


if __name__ == "__main__":
    export_raw_dataset()