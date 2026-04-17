import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import is_valid_step, preprocess_step


def preprocess_all_raw_data():
    input_file = Path("data/raw/raw_dataset_success_only.jsonl")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        return

    output_file = processed_dir / "processed_data.jsonl"
    kept = 0
    skipped = 0

    with open(input_file, "r", encoding="utf-8") as in_f, \
         open(output_file, "w", encoding="utf-8") as out_f:
        for line in in_f:
            step = json.loads(line)
            if is_valid_step(step):
                cleaned = preprocess_step(step)
                out_f.write(json.dumps(cleaned) + "\n")
                kept += 1
            else:
                skipped += 1

    print(f"Preprocessed: {kept} valid steps, {skipped} skipped")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    preprocess_all_raw_data()