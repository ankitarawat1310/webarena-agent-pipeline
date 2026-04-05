import json
from pathlib import Path


def save_episode(episode_data, output_path):
    """
    Append episode steps to a JSONL file.
    Each step becomes one line.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a", encoding="utf-8") as f:
        for step in episode_data:
            f.write(json.dumps(step) + "\n")

    print(f"Appended episode with {len(episode_data)} steps to: {output_path}")