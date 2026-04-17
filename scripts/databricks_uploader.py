import os
import json
import logging
import argparse
import requests
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class DatabricksClient:
    def __init__(self):
        self.host = os.environ["DATABRICKS_HOST"].rstrip("/")
        self.token = os.environ["DATABRICKS_TOKEN"]
        self.headers = {"Authorization": f"Bearer {self.token}"}
        self._ping()

    def _ping(self):
        r = requests.put(
            f"{self.host}/api/2.0/fs/files/Volumes/workspace/webarena/trajectories/.ping",
            headers=self.headers,
            data=b"ping",
            timeout=10,
        )
        if r.status_code not in (200, 201, 204):
            raise ConnectionError(f"Databricks connection failed: {r.status_code} {r.text[:200]}")
        log.info(f"Connected to Databricks: {self.host}")

    def upload(self, local: Path, volume_path: str) -> bool:
        size_kb = local.stat().st_size / 1024
        log.info(f"  Uploading {local.name} ({size_kb:.1f} KB) -> {volume_path}")
        with open(local, "rb") as fh:
            r = requests.put(
                f"{self.host}/api/2.0/fs/files/{volume_path.lstrip('/')}",
                headers=self.headers,
                data=fh,
                timeout=60,
            )
        if r.status_code not in (200, 201, 204):
            raise RuntimeError(f"Upload failed: {r.status_code} {r.text[:200]}")
        log.info(f"  Done: {local.name}")
        return True

    def upload_dir(self, local_dir: Path, volume_dir: str, glob: str = "*") -> list:
        uploaded = []
        for f in sorted(local_dir.glob(glob)):
            if f.is_file():
                volume_path = f"{volume_dir}/{f.name}"
                try:
                    self.upload(f, volume_path)
                    uploaded.append(volume_path)
                except Exception as e:
                    log.error(f"  Failed {f.name}: {e}")
        return uploaded


def upload_all(run_id: str, data_root: Path, volume_root: str = "/Volumes/workspace/webarena/trajectories") -> dict:
    client = DatabricksClient()
    base = f"{volume_root}/{run_id}"
    result = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "raw": [],
        "processed": [],
        "transformed": [],
        "splits": [],
    }

    raw_dir = data_root / "raw"
    if raw_dir.exists():
        log.info("[1/4] Uploading raw files...")
        result["raw"] = client.upload_dir(raw_dir, f"{base}/raw", "*.jsonl")

    proc_dir = data_root / "processed"
    if proc_dir.exists():
        log.info("[2/4] Uploading processed files...")
        result["processed"] = client.upload_dir(proc_dir, f"{base}/processed", "*.jsonl")

    trans_dir = data_root / "transformed"
    if trans_dir.exists():
        log.info("[3/4] Uploading transformed dataset...")
        result["transformed"] = client.upload_dir(trans_dir, f"{base}/transformed", "*.jsonl")

    splits_dir = data_root / "splits"
    if splits_dir.exists():
        log.info("[4/4] Uploading splits...")
        result["splits"] = client.upload_dir(splits_dir, f"{base}/splits", "*.jsonl")

    total = sum(len(v) for v in result.values() if isinstance(v, list))
    log.info(f"Upload complete: {total} files for run_id={run_id}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--volume-root", default="/Volumes/workspace/webarena/trajectories")
    args = parser.parse_args()

    upload_all(args.run_id, Path(args.data_root), args.volume_root)