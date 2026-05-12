#!/usr/bin/env python3
"""Generate publication-style plots from pipeline artifacts (compare, learning curves, replay)."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> object | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _agg_error_maps(agg: object) -> tuple[dict[str, dict[str, float]], bool]:
    """Build agent -> row map from compare_agents_aggregate.json."""
    if not isinstance(agg, list) or not agg:
        return {}, False
    err_map: dict[str, dict[str, float]] = {}
    for a in agg:
        if isinstance(a, dict) and "agent" in a:
            err_map[str(a["agent"])] = a  # type: ignore[assignment]
    return err_map, True


def _yerr_for_metric(err_map: dict[str, dict[str, float]], agents: list[str], metric: str) -> list[float] | None:
    """Resolve std column name; handles successful_steps_mean -> successful_steps_mean_std in aggregate."""
    std_candidates = [
        f"{metric}_std",
        "successful_steps_mean_std" if metric == "successful_steps_mean" else "",
    ]
    std_candidates = [s for s in std_candidates if s]
    yerr: list[float] = []
    for ag in agents:
        row = err_map.get(ag, {})
        found = None
        for sk in std_candidates:
            if sk in row:
                found = float(row[sk])
                break
        if found is None:
            return None
        yerr.append(found)
    return yerr


def plot_compare_metrics(
    reports_dir: Path,
    out_path: Path,
) -> bool:
    rows = _load_json(reports_dir / "compare_agents.json")
    if not isinstance(rows, list) or not rows:
        print(f"[plot_results] skip compare: missing or empty compare_agents.json")
        return False

    agg = _load_json(reports_dir / "compare_agents_aggregate.json")
    err_map, use_err = _agg_error_maps(agg)

    agents = [str(r["agent"]) for r in rows]
    x = np.arange(len(agents))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    metrics = [
        ("success_rate", "Success rate", 0.0, 1.05),
        ("invalid_action_rate", "Invalid action rate", 0.0, None),
        ("episode_latency_s", "Mean episode latency (s)", 0.0, None),
    ]

    for ax, (key, title, ymin, ymax) in zip(axes, metrics):
        vals = [float(r[key]) for r in rows]
        yerr = None
        if use_err:
            yerr = _yerr_for_metric(err_map, agents, key)
        ax.bar(x, vals, width=0.6, yerr=yerr, capsize=4, color="#4477AA", ecolor="#333333")
        ax.set_xticks(x)
        ax.set_xticklabels(agents, rotation=20, ha="right")
        ax.set_title(title)
        ax.set_ylabel(title.split("(")[0].strip())
        ax.set_ylim(bottom=ymin)
        if ymax is not None:
            ax.set_ylim(top=ymax)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Agent comparison (eval)", fontsize=12, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot_results] wrote {out_path}")
    return True


def plot_compare_efficiency(reports_dir: Path, out_path: Path) -> bool:
    """Return and step metrics for reporting (eval-time averages)."""
    rows = _load_json(reports_dir / "compare_agents.json")
    if not isinstance(rows, list) or not rows:
        print("[plot_results] skip compare_efficiency: missing compare_agents.json")
        return False

    agg = _load_json(reports_dir / "compare_agents_aggregate.json")
    err_map, use_err = _agg_error_maps(agg)

    agents = [str(r["agent"]) for r in rows]
    x = np.arange(len(agents))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    metrics = [
        ("mean_return", "Mean episodic return", None, None),
        (
            "steps_to_success",
            "Steps (mean; capped at max steps on failures)",
            0.0,
            None,
        ),
        ("successful_steps_mean", "Mean steps on successful episodes only", 0.0, None),
    ]

    for ax, (key, title, ymin, ymax) in zip(axes, metrics):
        vals = []
        for r in rows:
            if key not in r:
                vals.append(0.0)
            else:
                vals.append(float(r[key]))
        yerr = _yerr_for_metric(err_map, agents, key) if use_err else None
        ax.bar(x, vals, width=0.6, yerr=yerr, capsize=4, color="#AA7744", ecolor="#333333")
        ax.set_xticks(x)
        ax.set_xticklabels(agents, rotation=20, ha="right")
        ax.set_title(title)
        ax.set_ylabel("Value")
        if ymin is not None:
            ax.set_ylim(bottom=ymin)
        if ymax is not None:
            ax.set_ylim(top=ymax)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Eval efficiency and length (return includes shaping + terminal bonus)",
        fontsize=11,
        fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot_results] wrote {out_path}")
    return True


def plot_success_rate_by_seed(reports_dir: Path, out_path: Path) -> bool:
    """Heatmap of success rate per agent × seed when compare_agents_seed_*.json exists."""
    files = sorted(reports_dir.glob("compare_agents_seed_*.json"))
    if not files:
        print("[plot_results] skip success_by_seed: no compare_agents_seed_*.json")
        return False

    seed_rows: list[tuple[int, list[dict]]] = []
    for fp in files:
        m = re.search(r"compare_agents_seed_(\d+)\.json$", fp.name)
        if not m:
            continue
        data = _load_json(fp)
        if isinstance(data, list) and data:
            seed_rows.append((int(m.group(1)), data))

    if not seed_rows:
        return False

    seed_rows.sort(key=lambda t: t[0])
    seeds = [s for s, _ in seed_rows]
    agents = [str(r["agent"]) for r in seed_rows[0][1]]
    mat = np.zeros((len(agents), len(seeds)))
    for j, (_, data) in enumerate(seed_rows):
        by_agent = {str(r["agent"]): float(r.get("success_rate", 0.0)) for r in data}
        for i, ag in enumerate(agents):
            mat[i, j] = by_agent.get(ag, 0.0)

    fig, ax = plt.subplots(figsize=(max(6, len(seeds) * 1.2), 4), constrained_layout=True)
    im = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0, cmap="YlOrRd")
    ax.set_xticks(np.arange(len(seeds)))
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_yticks(np.arange(len(agents)))
    ax.set_yticklabels(agents)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Agent")
    ax.set_title("Success rate by seed (verifier)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Success rate")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot_results] wrote {out_path}")
    return True


def plot_return_vs_latency(reports_dir: Path, out_path: Path) -> bool:
    """Scatter: mean episodic return vs latency (annotated by agent)."""
    rows = _load_json(reports_dir / "compare_agents.json")
    if not isinstance(rows, list) or not rows:
        return False

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for r in rows:
        ag = str(r["agent"])
        ax.scatter(
            float(r.get("episode_latency_s", 0.0)),
            float(r.get("mean_return", 0.0)),
            s=120,
            label=ag,
        )
    ax.set_xlabel("Mean episode latency (s)")
    ax.set_ylabel("Mean episodic return (shaped)")
    ax.set_title("Return vs wall-clock cost per episode")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot_results] wrote {out_path}")
    return True


def plot_replay_sources(replay_path: Path, out_path: Path, max_lines: int = 50_000) -> bool:
    """Bar chart of transition counts by source tag (offline teacher mix)."""
    if not replay_path.exists():
        print(f"[plot_results] skip replay sources: {replay_path} not found")
        return False

    ctr: Counter[str] = Counter()
    with open(replay_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                src = str(obj.get("source", "unknown"))
                ctr[src] += 1
            except (json.JSONDecodeError, TypeError):
                continue

    if not ctr:
        print("[plot_results] skip replay sources: no rows parsed")
        return False

    labels = list(ctr.keys())
    counts = [ctr[k] for k in labels]
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, counts, color="#117733", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Transitions")
    ax.set_title(f"Replay buffer sources (n={sum(counts)} transitions read)")
    ax.grid(axis="x", alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot_results] wrote {out_path}")
    return True


def plot_learning_curves(reports_dir: Path, out_path: Path) -> bool:
    pattern = re.compile(r"learning_curve_(.+)_seed(\d+)\.json$")
    files = sorted(reports_dir.glob("learning_curve_*_seed*.json"))
    if not files:
        print("[plot_results] skip learning curves: no learning_curve_*.json")
        return False

    series: dict[str, list[dict]] = {}
    for fp in files:
        m = pattern.search(fp.name)
        if not m:
            continue
        agent = m.group(1)
        data = _load_json(fp)
        if isinstance(data, list):
            series[agent] = data

    if not series:
        return False

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    titles = ["TD loss (log scale)", "Epsilon", "Recent reward mean"]
    colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(series), 1)))

    for idx, (agent, points) in enumerate(sorted(series.items())):
        steps = [int(p["step"]) for p in points]
        color = colors[idx % len(colors)]
        loss = np.array([float(p["loss"]) for p in points], dtype=float)
        loss = np.clip(loss, 1e-12, None)
        axes[0].plot(steps, loss, label=agent, color=color, linewidth=1.5)
        axes[1].plot(steps, [float(p["epsilon"]) for p in points], label=agent, color=color, linewidth=1.5)
        axes[2].plot(steps, [float(p["recent_reward_mean"]) for p in points], label=agent, color=color, linewidth=1.5)

    axes[0].set_yscale("log")
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Training step")
    axes[2].set_ylabel("Recent episode return mean (shaped + terminal)")
    fig.suptitle(
        "Learning curves (recent reward is shaped; success uses verifier at eval)",
        fontsize=11,
        fontweight="bold",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot_results] wrote {out_path}")
    return True


def plot_replay_rewards(replay_path: Path, out_path: Path, max_lines: int = 20_000) -> bool:
    if not replay_path.exists():
        print(f"[plot_results] skip replay: {replay_path} not found")
        return False

    rewards: list[float] = []
    with open(replay_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rewards.append(float(obj.get("reward", 0.0)))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue

    if not rewards:
        print("[plot_results] skip replay: no reward values parsed")
        return False

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.hist(rewards, bins=min(40, max(10, len(set(rewards)))), color="#228833", edgecolor="white", alpha=0.85)
    ax.set_title(
        f"Replay reward distribution (n={len(rewards)} transitions; includes shaping offline/online)"
    )
    ax.set_xlabel("Reward")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot_results] wrote {out_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot WebArena RL experiment artifacts.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repo root (default: parent of scripts/).",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="Directory with compare_*.json and learning_curve_*.json.",
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=None,
        help="Path to transitions.jsonl (default: artifacts/replay/transitions.jsonl).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for PNGs (default: <project-root>/artifacts/reports/plots).",
    )
    args = parser.parse_args()

    root: Path = args.project_root
    reports_dir: Path = args.reports_dir or (root / "artifacts" / "reports")
    out_dir: Path = args.out_dir or (reports_dir / "plots")
    replay_path: Path = args.replay or (root / "artifacts" / "replay" / "transitions.jsonl")

    plot_compare_metrics(reports_dir, out_dir / "compare_metrics.png")
    plot_compare_efficiency(reports_dir, out_dir / "compare_efficiency.png")
    plot_success_rate_by_seed(reports_dir, out_dir / "compare_success_by_seed.png")
    plot_return_vs_latency(reports_dir, out_dir / "compare_return_vs_latency.png")
    plot_learning_curves(reports_dir, out_dir / "learning_curves.png")
    plot_replay_rewards(replay_path, out_dir / "replay_reward_distribution.png")
    plot_replay_sources(replay_path, out_dir / "replay_source_breakdown.png")


if __name__ == "__main__":
    main()
