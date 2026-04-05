import json

path = "data/raw/raw_dataset_success_only.jsonl"

episodes = set()
done_count = 0

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        episodes.add(row["episode_id"])
        if row.get("done") == True:
            done_count += 1

print("Unique episodes (trajectories):", len(episodes))
print("Completed trajectories (done=True):", done_count)