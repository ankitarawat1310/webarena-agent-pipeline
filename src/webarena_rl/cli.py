from __future__ import annotations

import argparse
import json
from statistics import mean, pstdev
from pathlib import Path

import yaml

from .config import load_config
from .training import run_compare, run_offline_prefill, run_train


def main() -> None:
    parser = argparse.ArgumentParser(description="WebArena RL baseline CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prefill = sub.add_parser("offline-prefill")
    p_prefill.add_argument("--config", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--config", required=True)
    p_train.add_argument("--agent", required=True, choices=["vanilla_dqn", "ddqn"])
    p_train.add_argument("--seed", type=int, default=None)

    p_compare = sub.add_parser("compare")
    p_compare.add_argument("--config", required=True)

    args = parser.parse_args()
    app_cfg = load_config(args.config)
    cfg = app_cfg.raw
    seed = app_cfg.project_seed
    train_tasks, eval_tasks = _load_task_bank(cfg)

    if args.cmd == "offline-prefill":
        tasks = train_tasks or [
            "Find a white desk under $200 and open details.",
            "Search for gaming keyboard and sort by low price.",
            "Add a laptop sleeve to cart.",
        ]
        count = run_offline_prefill(cfg, tasks=tasks, seed=seed)
        print(f"offline_prefill_transitions={count}")
        return

    if args.cmd == "train":
        run_seed = int(args.seed) if args.seed is not None else seed
        result = run_train(cfg, agent=args.agent, seed=run_seed, tasks=train_tasks)
        print(json.dumps(result.__dict__, ensure_ascii=True))
        return

    if args.cmd == "compare":
        seeds = [int(s) for s in cfg.get("project", {}).get("seeds", [seed])]
        all_seed_rows: dict[int, list[dict[str, float | str]]] = {}
        out_path = Path("artifacts/reports/compare_agents.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        for run_seed in seeds:
            rows = run_compare(cfg, seed=run_seed, tasks=eval_tasks)
            all_seed_rows[run_seed] = rows
            with open(out_path.parent / f"compare_agents_seed_{run_seed}.json", "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2, ensure_ascii=True)
        aggregate_rows = _aggregate_compare(all_seed_rows)
        with open(out_path.parent / "compare_agents_aggregate.json", "w", encoding="utf-8") as f:
            json.dump(aggregate_rows, f, indent=2, ensure_ascii=True)
        rows = all_seed_rows[seeds[0]]
        out_path = Path("artifacts/reports/compare_agents.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=True)
        print(json.dumps({"seeds": seeds, "aggregate": aggregate_rows}, indent=2, ensure_ascii=True))


def _load_task_bank(cfg: dict) -> tuple[list[str], list[str]]:
    tasks_cfg = cfg.get("tasks", {})
    train_path = tasks_cfg.get("train_path")
    eval_path = tasks_cfg.get("eval_path")

    def _read(path: str | None) -> list[str]:
        if not path:
            return []
        p = Path(path)
        if not p.exists():
            return []
        with open(p, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if isinstance(raw, dict):
            return [str(x) for x in raw.get("tasks", [])]
        if isinstance(raw, list):
            return [str(x) for x in raw]
        return []

    return _read(train_path), _read(eval_path)


def _aggregate_compare(all_seed_rows: dict[int, list[dict[str, float | str]]]) -> list[dict[str, float | str]]:
    metric_names = [
        "success_rate",
        "mean_return",
        "steps_to_success",
        "successful_steps_mean",
        "invalid_action_rate",
        "episode_latency_s",
    ]
    by_agent: dict[str, dict[str, list[float]]] = {}
    for rows in all_seed_rows.values():
        for row in rows:
            agent = str(row["agent"])
            bucket = by_agent.setdefault(agent, {metric: [] for metric in metric_names})
            for metric in metric_names:
                if metric in row:
                    bucket[metric].append(float(row[metric]))
    aggregated: list[dict[str, float | str]] = []
    for agent, metrics in by_agent.items():
        out: dict[str, float | str] = {"agent": agent}
        for metric, values in metrics.items():
            if not values:
                continue
            out[f"{metric}_mean"] = round(mean(values), 4)
            out[f"{metric}_std"] = round(pstdev(values), 4) if len(values) > 1 else 0.0
        aggregated.append(out)
    return aggregated


if __name__ == "__main__":
    main()

