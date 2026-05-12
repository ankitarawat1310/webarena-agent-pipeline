from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def main() -> None:
    path = Path("artifacts/replay/transitions.jsonl")
    if not path.exists():
        print("replay file missing:", path)
        return
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print("transitions:", len(rows))
    if not rows:
        return
    rewards = [float(r.get("reward", 0.0)) for r in rows]
    reward_bins = Counter()
    for r in rewards:
        if r < 0:
            reward_bins["negative"] += 1
        elif r > 0:
            reward_bins["positive"] += 1
        else:
            reward_bins["zero"] += 1
    print("reward_counts:", dict(reward_bins))
    print("positive_rewards:", sum(1 for r in rewards if r > 0))
    print("terminal_transitions:", sum(1 for r in rows if bool(r.get("done", False))))
    print(
        "verified_success_transitions:",
        sum(1 for r in rows if bool(((r.get("metadata") or {}).get("verifier") or {}).get("verified_success", False))),
    )
    print("source_counts:", dict(Counter(str(r.get("source", "unknown")) for r in rows)))
    action_type_counts = Counter(str((r.get("action") or {}).get("action_type", "unknown")) for r in rows)
    print("action_type_counts:", dict(action_type_counts))
    clicked_texts = Counter()
    clicked_hrefs = Counter()
    for row in rows:
        action = row.get("action") or {}
        if action.get("action_type") != "click":
            continue
        state = row.get("state") or {}
        element_id = int(action.get("element_id", 0))
        for el in state.get("elements", []):
            if int(el.get("id", -1)) != element_id:
                continue
            text = str(el.get("text", "")).strip()
            href = str(el.get("href", "")).strip()
            if text:
                clicked_texts[text] += 1
            if href:
                clicked_hrefs[href] += 1
            break
    print("top_clicked_texts:", clicked_texts.most_common(10))
    print("top_clicked_hrefs:", clicked_hrefs.most_common(10))
    if reward_bins.get("positive", 0) == 0:
        print("WARNING: replay has no positive rewards")
    if action_type_counts.get("stop", 0) == 0:
        print("WARNING: replay has no stop actions")


if __name__ == "__main__":
    main()
