from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AppConfig:
    raw: dict[str, Any]

    @property
    def project_seed(self) -> int:
        return int(self.raw["project"]["seed"])

    @property
    def replay_path(self) -> Path:
        return Path(self.raw["replay"]["path"])


def load_config(path: str | Path) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig(raw=raw)

