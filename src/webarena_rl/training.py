from __future__ import annotations

import json
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import WebArenaShoppingEnv
from .llm_policy import PromptConfig, SmallLlmPromptPolicy
from .replay import JsonlReplayStore, Transition
from .scripted_policy import ScriptedShoppingPolicy
from .task_utils import extract_search_query


@dataclass
class TrainResult:
    agent: str
    episodes: int
    avg_reward: float


class QNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def run_offline_prefill(cfg: dict[str, Any], tasks: list[str], seed: int) -> int:
    _seed_everything(seed)
    rng = random.Random(seed)
    debug = bool(cfg.get("debug", {}).get("enabled", False))
    trace_dir = Path(cfg.get("debug", {}).get("trace_dir", "artifacts/reports/traces"))
    trace_dir.mkdir(parents=True, exist_ok=True)
    replay = JsonlReplayStore(path=_replay_path(cfg))
    env = WebArenaShoppingEnv(
        shopping_url=cfg["environment"]["shopping_url"],
        admin_url=cfg["environment"]["admin_url"],
        max_steps=cfg["environment"]["max_steps_per_episode"],
        seed=seed,
        headless=bool(cfg["environment"].get("headless", True)),
        debug=debug,
        trace_path=str(trace_dir / f"offline_prefill_env_seed{seed}.jsonl"),
        repair_invalid_actions=bool(cfg["environment"].get("repair_invalid_actions", False)),
        clear_state_on_reset=bool(cfg["environment"].get("clear_state_on_reset", True)),
        force_click=bool(cfg["environment"].get("force_click", False)),
        reset_url=cfg["environment"].get("reset_url"),
    )
    env.health_check()
    llm = SmallLlmPromptPolicy(
        PromptConfig(
            model_name=str(cfg["llm_baseline"]["model_name"]),
            temperature=float(cfg["llm_baseline"]["temperature"]),
            top_p=float(cfg["llm_baseline"]["top_p"]),
            max_tokens=int(cfg["llm_baseline"]["max_tokens"]),
            history_window=int(cfg["llm_baseline"]["history_window"]),
            retry_invalid_json=int(cfg["llm_baseline"]["retry_invalid_json"]),
            use_real_llm=bool(cfg["llm_baseline"].get("use_real_llm", False)),
            backend=str(cfg["llm_baseline"].get("backend", "ollama")),
            endpoint=str(cfg["llm_baseline"].get("endpoint", "http://localhost:11434")),
            timeout_seconds=float(cfg["llm_baseline"].get("timeout_seconds", 30.0)),
        ),
        seed=seed,
        debug=debug,
        trace_path=str(trace_dir / f"offline_prefill_llm_seed{seed}.jsonl"),
    )
    scripted = ScriptedShoppingPolicy()
    mix = cfg.get("offline_prefill", {}).get("teacher_mix", {"scripted": 0.6, "llm": 0.3, "random": 0.1})
    max_episodes = int(cfg.get("offline_prefill", {}).get("max_episodes", 500))
    target = int(cfg["replay"]["offline_prefill_target_transitions"])
    written = 0
    success_count = 0

    def _sample_teacher() -> str:
        names = ["scripted", "llm", "random"]
        weights = [float(mix.get("scripted", 0.6)), float(mix.get("llm", 0.3)), float(mix.get("random", 0.1))]
        return rng.choices(names, weights=weights, k=1)[0]

    try:
        for episode in range(max_episodes):
            if written >= target:
                break
            task = tasks[episode % max(len(tasks), 1)] if tasks else "Find a product under $100 and open details."
            obs = env.reset(task)
            teacher = _sample_teacher()
            done = False
            while not done and written < target:
                templates = build_action_templates(int(cfg["training"].get("max_grounded_elements", 30)))
                if teacher == "scripted":
                    action = scripted.act(obs)
                    source = "scripted_offline"
                elif teacher == "llm":
                    action = llm.act(obs)
                    source = "llm_offline"
                else:
                    mask = valid_action_mask(obs, templates)
                    idx = select_action_idx(np.zeros(len(templates), dtype=np.float32), mask, epsilon=1.0, rng=rng)
                    action = decode_action(idx, obs, templates)
                    source = "random_offline"
                step = env.step(action)
                replay.append(
                    Transition(
                        state=obs,
                        action=action,
                        reward=step.reward,
                        next_state=step.observation,
                        done=step.done,
                        metadata={"task": task, "verifier": step.info.get("verifier"), "episode": episode},
                        source=source,
                    )
                )
                if step.info.get("verified_success", False):
                    success_count += 1
                obs = step.observation
                done = step.done
                written += 1
    finally:
        env.close()
    print(f"[offline_prefill] transitions={written} verified_successes={success_count}", flush=True)
    return written


def run_train(cfg: dict[str, Any], agent: str, seed: int, tasks: list[str] | None = None) -> TrainResult:
    _seed_everything(seed)
    rng = random.Random(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(cfg["training"]["batch_size"])
    gamma = float(cfg["training"]["gamma"])
    lr = float(cfg["training"]["learning_rate"])
    sync_steps = int(cfg["training"]["target_sync_steps"])
    total_steps = int(cfg["training"]["total_steps"])
    eps_start = float(cfg["training"]["epsilon_start"])
    eps_final = float(cfg["training"]["epsilon_final"])
    eps_decay = int(cfg["training"]["epsilon_decay_steps"])
    grad_clip = float(cfg["training"]["grad_clip_norm"])
    max_grounded_elems = int(cfg["training"].get("max_grounded_elements", 30))
    action_templates = build_action_templates(max_grounded_elems)
    debug = bool(cfg.get("debug", {}).get("enabled", False))
    trace_dir = Path(cfg.get("debug", {}).get("trace_dir", "artifacts/reports/traces"))
    trace_dir.mkdir(parents=True, exist_ok=True)
    encoder_fn, obs_dim = _build_obs_encoder(cfg)

    online = QNet(obs_dim, len(action_templates)).to(device)
    target = QNet(obs_dim, len(action_templates)).to(device)
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=lr)
    env = WebArenaShoppingEnv(
        shopping_url=cfg["environment"]["shopping_url"],
        admin_url=cfg["environment"]["admin_url"],
        max_steps=cfg["environment"]["max_steps_per_episode"],
        seed=seed,
        headless=bool(cfg["environment"].get("headless", True)),
        debug=debug,
        trace_path=str(trace_dir / f"train_env_{agent}_seed{seed}.jsonl"),
        repair_invalid_actions=bool(cfg["environment"].get("repair_invalid_actions", False)),
        clear_state_on_reset=bool(cfg["environment"].get("clear_state_on_reset", True)),
        force_click=bool(cfg["environment"].get("force_click", False)),
        reset_url=cfg["environment"].get("reset_url"),
    )
    env.health_check()
    offline_rows = _load_replay(_replay_path(cfg))
    online_buffer = deque(maxlen=int(cfg["replay"].get("online_capacity", 50000)))
    if not offline_rows:
        print("[train] warning: offline replay is empty; training from online only", flush=True)
    episode_rewards: list[float] = []
    curve_points: list[dict[str, Any]] = []
    invalid_replay_actions = 0
    episode_reward = 0.0
    task_pool = tasks or [f"Find a relevant product and open details ({agent})."]
    obs = env.reset(rng.choice(task_pool))
    try:
        for step_idx in range(1, total_steps + 1):
            epsilon = eps_final + (eps_start - eps_final) * np.exp(-step_idx / max(eps_decay, 1))
            with torch.no_grad():
                q_vals = online(_obs_to_tensor(obs, device, encoder_fn)).squeeze(0).cpu().numpy()
            mask = valid_action_mask(obs, action_templates)
            action_idx = select_action_idx(q_vals, mask, epsilon=epsilon, rng=rng)
            action = decode_action(action_idx, obs, action_templates)
            prev_obs = obs
            env_step = env.step(action)
            online_buffer.append(
                {
                    "state": prev_obs,
                    "action": action,
                    "reward": env_step.reward,
                    "next_state": env_step.observation,
                    "done": env_step.done,
                    "metadata": {"task": prev_obs.get("instruction", ""), "step": step_idx, "info": env_step.info},
                    "source": "online_env",
                }
            )
            episode_reward += env_step.reward
            obs = env_step.observation
            if env_step.done:
                episode_rewards.append(episode_reward)
                episode_reward = 0.0
                obs = env.reset(rng.choice(task_pool))
            batch = sample_mixed_batch(
                offline_rows=offline_rows,
                online_rows=list(online_buffer),
                batch_size=batch_size,
                offline_fraction=float(cfg["replay"].get("offline_to_online_sample_ratio", 0.5)),
                rng=rng,
            )
            loss, invalid_count = _compute_td_loss(
                batch=batch,
                online=online,
                target=target,
                gamma=gamma,
                device=device,
                ddqn=(agent == "ddqn"),
                encoder_fn=encoder_fn,
                action_templates=action_templates,
            )
            invalid_replay_actions += invalid_count
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), grad_clip)
            opt.step()
            if step_idx % 100 == 0:
                curve_points.append(
                    {
                        "step": step_idx,
                        "epsilon": float(epsilon),
                        "loss": float(loss.item()),
                        "recent_reward_mean": float(np.mean(episode_rewards[-20:])) if episode_rewards else 0.0,
                    }
                )
            if step_idx % sync_steps == 0:
                target.load_state_dict(online.state_dict())
    finally:
        env.close()
    if invalid_replay_actions:
        print(f"[train] replay_action_mapping_fallbacks={invalid_replay_actions}", flush=True)
    out_dir = Path("artifacts/models")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(online.state_dict(), out_dir / f"{agent}.pt")
    curve_out = Path("artifacts/reports")
    curve_out.mkdir(parents=True, exist_ok=True)
    with open(curve_out / f"learning_curve_{agent}_seed{seed}.json", "w", encoding="utf-8") as f:
        json.dump(curve_points, f, ensure_ascii=True, indent=2)
    avg_reward = float(np.mean(episode_rewards[-20:])) if episode_rewards else 0.0
    return TrainResult(agent=agent, episodes=len(episode_rewards), avg_reward=avg_reward)


def run_compare(cfg: dict[str, Any], seed: int, tasks: list[str] | None = None) -> list[dict[str, Any]]:
    _seed_everything(seed)
    rng = random.Random(seed)
    eval_episodes = int(cfg["comparison"]["eval_episodes"])
    agents = list(cfg["comparison"]["agents"])
    total_agents = len(agents)
    total_episodes_all_agents = max(total_agents * eval_episodes, 1)
    reports_dir = Path("artifacts/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = reports_dir / f"compare_episodes_seed{seed}.jsonl"
    if episodes_path.exists():
        episodes_path.unlink()
    status_path = reports_dir / "compare_status.json"
    debug = bool(cfg.get("debug", {}).get("enabled", False))
    trace_dir = Path(cfg.get("debug", {}).get("trace_dir", "artifacts/reports/traces"))
    trace_dir.mkdir(parents=True, exist_ok=True)
    env = WebArenaShoppingEnv(
        shopping_url=cfg["environment"]["shopping_url"],
        admin_url=cfg["environment"]["admin_url"],
        max_steps=cfg["environment"]["max_steps_per_episode"],
        seed=seed,
        headless=bool(cfg["environment"].get("headless", True)),
        debug=debug,
        trace_path=str(trace_dir / f"compare_env_seed{seed}.jsonl"),
        repair_invalid_actions=bool(cfg["environment"].get("repair_invalid_actions", False)),
        clear_state_on_reset=bool(cfg["environment"].get("clear_state_on_reset", True)),
        force_click=bool(cfg["environment"].get("force_click", False)),
        reset_url=cfg["environment"].get("reset_url"),
    )
    env.health_check()
    task_pool = tasks or ["Find an item and navigate toward checkout."]
    episode_tasks = [task_pool[i % len(task_pool)] for i in range(eval_episodes)]
    rng.shuffle(episode_tasks)
    _write_compare_status(status_path, {"phase": "starting", "current_agent": None, "agent_index": 0, "total_agents": total_agents, "current_episode": 0, "episodes_per_agent": eval_episodes, "overall_episode": 0, "overall_total_episodes": total_episodes_all_agents, "progress_pct": 0.0, "updated_at_epoch_s": round(time.time(), 3)})
    results: list[dict[str, Any]] = []
    try:
        for agent_idx, agent in enumerate(agents, start=1):
            policy = _policy_from_agent_name(cfg, agent, seed)
            success = 0
            total_steps = 0
            invalid_actions = 0
            returns: list[float] = []
            success_steps: list[int] = []
            start = time.time()
            for episode_idx, task in enumerate(episode_tasks, start=1):
                obs = env.reset(task)
                ep_return = 0.0
                ep_steps = 0
                ep_invalid = 0
                done = False
                episode_start = time.time()
                final_info: dict[str, Any] = {}
                while not done:
                    action = policy(obs)
                    step = env.step(action)
                    ep_steps += 1
                    ep_return += step.reward
                    if not step.info.get("action_ok", True):
                        ep_invalid += 1
                    final_info = step.info
                    obs = step.observation
                    done = step.done
                verified_success = bool(final_info.get("verified_success") or (final_info.get("verifier") or {}).get("verified_success", False))
                success += int(verified_success)
                if verified_success:
                    success_steps.append(ep_steps)
                total_steps += ep_steps
                invalid_actions += ep_invalid
                returns.append(ep_return)
                with open(episodes_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"agent": agent, "seed": seed, "episode": episode_idx, "task": task, "success": verified_success, "steps": ep_steps, "return": ep_return, "invalid_actions": ep_invalid, "final_url": str(obs.get("page", "")), "latency_s": round(time.time() - episode_start, 3)}, ensure_ascii=True) + "\n")
                overall_episode = (agent_idx - 1) * eval_episodes + episode_idx
                _write_compare_status(status_path, {"phase": "running", "current_agent": agent, "agent_index": agent_idx, "total_agents": total_agents, "current_episode": episode_idx, "episodes_per_agent": eval_episodes, "overall_episode": overall_episode, "overall_total_episodes": total_episodes_all_agents, "progress_pct": round(100.0 * overall_episode / total_episodes_all_agents, 2), "updated_at_epoch_s": round(time.time(), 3)})
            elapsed = time.time() - start
            failures = eval_episodes - success
            capped_steps = list(success_steps) + [int(cfg["environment"]["max_steps_per_episode"])] * max(failures, 0)
            agent_result = {
                "agent": agent,
                "success_rate": round(success / max(eval_episodes, 1), 4),
                "mean_return": round(float(np.mean(returns)) if returns else 0.0, 4),
                "steps_to_success": round(float(np.mean(capped_steps)) if capped_steps else 0.0, 2),
                "successful_steps_mean": round(float(np.mean(success_steps)) if success_steps else 0.0, 2),
                "invalid_action_rate": round(invalid_actions / max(total_steps, 1), 4),
                "episode_latency_s": round(elapsed / max(eval_episodes, 1), 2),
            }
            results.append(agent_result)
    finally:
        env.close()
    _write_compare_status(status_path, {"phase": "completed", "current_agent": None, "agent_index": total_agents, "total_agents": total_agents, "current_episode": eval_episodes, "episodes_per_agent": eval_episodes, "overall_episode": total_agents * eval_episodes, "overall_total_episodes": total_episodes_all_agents, "progress_pct": 100.0, "results": results, "updated_at_epoch_s": round(time.time(), 3)})
    return results


def _write_compare_status(status_path: Path, status: dict[str, Any]) -> None:
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=True, indent=2)


def _replay_path(cfg: dict[str, Any]):
    return Path(cfg["replay"]["path"])


def _obs_to_vec(obs: dict[str, Any]) -> np.ndarray:
    page = str(obs.get("page", "")).lower()
    elements = obs.get("elements", [])
    history = obs.get("history", [])
    text_blob = " ".join(str(e.get("text", "")).lower() for e in elements[:20])
    return np.array(
        [
            len(elements) / 50.0,
            len(history) / 10.0,
            1.0 if "cart" in page else 0.0,
            1.0 if "search" in page else 0.0,
            1.0 if "checkout" in page else 0.0,
            1.0 if "add to cart" in text_blob else 0.0,
            1.0 if "price" in text_blob else 0.0,
            1.0,
        ],
        dtype=np.float32,
    )


def _obs_to_tensor(obs: dict[str, Any], device: torch.device, encoder_fn: Callable[[dict[str, Any]], np.ndarray]) -> torch.Tensor:
    arr = encoder_fn(obs)
    return torch.tensor(arr, device=device).unsqueeze(0)


def _load_replay(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def sample_mixed_batch(
    offline_rows: list[dict[str, Any]],
    online_rows: list[dict[str, Any]],
    batch_size: int,
    offline_fraction: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if not offline_rows and not online_rows:
        raise RuntimeError("No replay data available")
    if not online_rows:
        return _sample_rows(offline_rows, batch_size, rng)
    if not offline_rows:
        return _sample_rows(online_rows, batch_size, rng)
    offline_n = int(round(batch_size * max(0.0, min(1.0, offline_fraction))))
    online_n = batch_size - offline_n
    return _sample_rows(offline_rows, offline_n, rng) + _sample_rows(online_rows, online_n, rng)


def _sample_rows(rows: list[dict[str, Any]], n: int, rng: random.Random) -> list[dict[str, Any]]:
    if n <= 0:
        return []
    if len(rows) >= n:
        return rng.sample(rows, n)
    return [rows[rng.randrange(len(rows))] for _ in range(n)]


def _compute_td_loss(
    batch: list[dict[str, Any]],
    online: QNet,
    target: QNet,
    gamma: float,
    device: torch.device,
    ddqn: bool,
    encoder_fn: Callable[[dict[str, Any]], np.ndarray],
    action_templates: list[tuple[str, int | None]],
) -> tuple[torch.Tensor, int]:
    states = torch.tensor(np.stack([encoder_fn(r["state"]) for r in batch]), device=device)
    action_indices: list[int] = []
    invalid_count = 0
    for row in batch:
        idx = action_to_idx(row["action"], row["state"], action_templates)
        if idx == _wait_idx(action_templates) and str(row["action"].get("action_type", "wait")) != "wait":
            invalid_count += 1
        action_indices.append(idx)
    actions = torch.tensor(action_indices, dtype=torch.int64, device=device)
    rewards = torch.tensor([float(r["reward"]) for r in batch], dtype=torch.float32, device=device)
    next_states = torch.tensor(np.stack([encoder_fn(r["next_state"]) for r in batch]), device=device)
    dones = torch.tensor([float(r["done"]) for r in batch], dtype=torch.float32, device=device)

    q = online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_masks = np.stack([valid_action_mask(row["next_state"], action_templates) for row in batch])
        next_mask_t = torch.tensor(next_masks, dtype=torch.bool, device=device)
        if ddqn:
            online_next_q = online(next_states).masked_fill(~next_mask_t, -1e9)
            next_actions = torch.argmax(online_next_q, dim=1)
            target_next_q = target(next_states).masked_fill(~next_mask_t, -1e9)
            next_q = target_next_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        else:
            next_q = torch.max(target(next_states).masked_fill(~next_mask_t, -1e9), dim=1).values
        target_q = rewards + gamma * (1.0 - dones) * next_q
    return F.mse_loss(q, target_q), invalid_count


def _policy_from_agent_name(cfg: dict[str, Any], agent: str, seed: int):
    if agent == "llm_prompt_baseline":
        llm = SmallLlmPromptPolicy(
            PromptConfig(
                model_name=str(cfg["llm_baseline"]["model_name"]),
                temperature=float(cfg["llm_baseline"]["temperature"]),
                top_p=float(cfg["llm_baseline"]["top_p"]),
                max_tokens=int(cfg["llm_baseline"]["max_tokens"]),
                history_window=int(cfg["llm_baseline"]["history_window"]),
                retry_invalid_json=int(cfg["llm_baseline"]["retry_invalid_json"]),
                use_real_llm=bool(cfg["llm_baseline"].get("use_real_llm", False)),
                backend=str(cfg["llm_baseline"].get("backend", "ollama")),
                endpoint=str(cfg["llm_baseline"].get("endpoint", "http://localhost:11434")),
                timeout_seconds=float(cfg["llm_baseline"].get("timeout_seconds", 30.0)),
            ),
            seed=seed,
        )
        return llm.act

    model_path = Path("artifacts/models") / f"{agent}.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found for agent '{agent}': {model_path}. "
            "Train the agent before running compare."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_fn, obs_dim = _build_obs_encoder(cfg)
    action_templates = build_action_templates(int(cfg["training"].get("max_grounded_elements", 30)))
    model = QNet(obs_dim, len(action_templates)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    def _act(obs: dict[str, Any]) -> dict[str, Any]:
        with torch.no_grad():
            q = model(_obs_to_tensor(obs, device, encoder_fn)).squeeze(0).cpu().numpy()
            idx = select_action_idx(q, valid_action_mask(obs, action_templates), epsilon=0.0, rng=random.Random(seed))
        return decode_action(idx, obs, action_templates)

    return _act


def _build_obs_encoder(cfg: dict[str, Any]) -> tuple[Callable[[dict[str, Any]], np.ndarray], int]:
    encoder_name = str(cfg.get("training", {}).get("state_encoder", "handcrafted_8d")).lower()
    if encoder_name != "sentence_transformer":
        return _obs_to_vec, 8

    model_name = str(cfg.get("training", {}).get("sentence_transformer_model", "sentence-transformers/all-MiniLM-L6-v2"))
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:  # noqa: BLE001
        print("[encoder] sentence-transformers not installed; falling back to handcrafted_8d", flush=True)
        return _obs_to_vec, 8

    model = SentenceTransformer(model_name)
    if hasattr(model, "get_embedding_dimension"):
        sample_dim = int(model.get_embedding_dimension())
    else:
        sample_dim = int(model.get_sentence_embedding_dimension())

    def _encode(obs: dict[str, Any]) -> np.ndarray:
        instruction = str(obs.get("instruction", ""))
        page = str(obs.get("page", ""))
        elements = obs.get("elements", [])
        history = obs.get("history", [])
        top_text = " ".join(str(e.get("text", "")) for e in elements[:20])
        hist = " ".join(str(h.get("action_type", "")) for h in history[-10:])
        text = f"instruction: {instruction}\npage: {page}\nelements: {top_text}\nhistory: {hist}"
        emb = model.encode(text, normalize_embeddings=True)
        return np.array(emb, dtype=np.float32)

    return _encode, sample_dim


def build_action_templates(max_elems: int) -> list[tuple[str, int | None]]:
    templates: list[tuple[str, int | None]] = []
    templates.extend(("click", slot) for slot in range(max_elems))
    templates.extend(("type", slot) for slot in range(max_elems))
    templates.extend([("scroll", None), ("go_back", None), ("wait", None), ("stop", None)])
    return templates


def valid_action_mask(obs: dict[str, Any], action_templates: list[tuple[str, int | None]]) -> np.ndarray:
    elements = list(obs.get("elements", []))
    clickable = [e for e in elements if _is_valid_click_target(e)]
    typable = [e for e in elements if _is_valid_type_target(e)]
    mask = np.zeros(len(action_templates), dtype=bool)
    for i, (name, slot) in enumerate(action_templates):
        if name == "click":
            mask[i] = slot is not None and slot < len(clickable)
        elif name == "type":
            mask[i] = slot is not None and slot < len(typable)
        else:
            mask[i] = True
    return mask


def select_action_idx(q_values: np.ndarray, mask: np.ndarray, epsilon: float, rng: random.Random) -> int:
    valid_idxs = [i for i, ok in enumerate(mask.tolist()) if ok]
    if not valid_idxs:
        return 0
    if rng.random() < epsilon:
        return rng.choice(valid_idxs)
    masked_q = np.where(mask, q_values, -1e9)
    return int(np.argmax(masked_q))


def decode_action(action_idx: int, obs: dict[str, Any], action_templates: list[tuple[str, int | None]]) -> dict[str, Any]:
    action_type, slot = action_templates[action_idx]
    if action_type in {"scroll", "go_back", "wait", "stop"}:
        return {"action_type": action_type, "element_id": 0, "text": "", "key": ""}
    if action_type == "click":
        clickable = [e for e in obs.get("elements", []) if _is_valid_click_target(e)]
        if slot is None or slot >= len(clickable):
            return {"action_type": "wait", "element_id": 0, "text": "", "key": ""}
        return {"action_type": "click", "element_id": int(clickable[slot]["id"]), "text": "", "key": ""}
    typable = [e for e in obs.get("elements", []) if _is_valid_type_target(e)]
    if slot is None or slot >= len(typable):
        return {"action_type": "wait", "element_id": 0, "text": "", "key": ""}
    return {
        "action_type": "type",
        "element_id": int(typable[slot]["id"]),
        "text": extract_search_query(str(obs.get("instruction", ""))),
        "key": "",
    }


def action_to_idx(action: dict[str, Any], state: dict[str, Any], action_templates: list[tuple[str, int | None]]) -> int:
    action_type = str(action.get("action_type", "wait")).lower()
    if action_type in {"scroll", "go_back", "wait", "stop"}:
        return action_templates.index((action_type, None))
    element_id = int(action.get("element_id", 0))
    if action_type == "click":
        clickable = [e for e in state.get("elements", []) if _is_valid_click_target(e)]
        for slot, el in enumerate(clickable):
            if int(el.get("id", -1)) == element_id:
                return action_templates.index(("click", slot))
    if action_type == "type":
        typable = [e for e in state.get("elements", []) if _is_valid_type_target(e)]
        for slot, el in enumerate(typable):
            if int(el.get("id", -1)) == element_id:
                return action_templates.index(("type", slot))
    return _wait_idx(action_templates)


def _wait_idx(action_templates: list[tuple[str, int | None]]) -> int:
    return action_templates.index(("wait", None))


def _is_valid_click_target(el: dict[str, Any]) -> bool:
    if not bool(el.get("visible", False)) or not bool(el.get("enabled", False)):
        return False
    if bool(el.get("is_text_input", False)):
        return False
    text = str(el.get("text", "")).lower()
    href = str(el.get("href", "")).lower().strip()
    cls = str(el.get("class_name", "")).lower()
    if "skip to" in text or "skip" in cls:
        return False
    if href.startswith("#"):
        return False
    role = str(el.get("role", "")).lower()
    tag = str(el.get("tag", "")).lower()
    return tag in {"a", "button", "input"} or "button" in role or "link" in role


def _is_valid_type_target(el: dict[str, Any]) -> bool:
    return bool(el.get("visible", False)) and bool(el.get("enabled", False)) and bool(el.get("is_text_input", False))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

