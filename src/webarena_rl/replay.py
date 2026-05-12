from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class Transition:
    state: dict[str, Any]
    action: dict[str, Any]
    reward: float
    next_state: dict[str, Any]
    done: bool
    metadata: dict[str, Any]
    source: str  # llm_offline | online_env


class JsonlReplayStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, transition: Transition) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(transition), ensure_ascii=True) + "\n")

