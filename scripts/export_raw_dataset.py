import json
from pathlib import Path


def export_raw_dataset():
    raw_dir = Path("data/raw")
    input_file = raw_dir / "raw_dataset_success_only.jsonl"
    output_file = Path("data/raw_dataset.jsonl")

    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        return

    total_records = 0

    with open(input_file, "r", encoding="utf-8") as in_f, \
         open(output_file, "w", encoding="utf-8") as out_f:
        for line in in_f:
            out_f.write(line)
            total_records += 1

    print(f"Created combined dataset: {output_file}")
    print(f"Total records: {total_records}")


if __name__ == "__main__":
    export_raw_dataset()