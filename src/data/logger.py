import json
from pathlib import Path


def save_episode(episode_data, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(episode_data, f, indent=2)

    print(f"Saved episode to: {output_path}")