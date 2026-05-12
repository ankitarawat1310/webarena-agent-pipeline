from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from .task_utils import extract_search_query


PROMPT_SYSTEM = (
    "You are a web shopping assistant. Output exactly one valid JSON action object "
    "with keys: action_type, element_id, text, key."
)


@dataclass
class PromptConfig:
    model_name: str
    temperature: float
    top_p: float
    max_tokens: int
    history_window: int
    retry_invalid_json: int
    use_real_llm: bool
    backend: str
    endpoint: str
    timeout_seconds: float


class SmallLlmPromptPolicy:
    """Prompt-policy baseline with strict output contract.

    Uses a deterministic heuristic fallback so benchmarking remains runnable
    without external model serving dependencies.
    """

    def __init__(self, cfg: PromptConfig, seed: int, debug: bool = False, trace_path: str | None = None) -> None:
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.debug = debug
        self.trace_path = Path(trace_path) if trace_path else None
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)

    def build_user_prompt(self, observation: dict[str, Any]) -> str:
        history = observation.get("history", [])[-self.cfg.history_window :]
        payload = {
            "instruction": observation.get("instruction", ""),
            "page": observation.get("page", ""),
            "elements": observation.get("elements", []),
            "history": history,
            "output_schema": "strict_json_action_v1",
        }
        return json.dumps(payload, ensure_ascii=True)

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        user_prompt = self.build_user_prompt(observation)
        for _attempt in range(self.cfg.retry_invalid_json + 1):
            raw = self._generate_raw_action(observation, user_prompt)
            try:
                action = json.loads(raw)
                parsed = {
                    "action_type": action.get("action_type", "click"),
                    "element_id": int(action.get("element_id", 1)),
                    "text": str(action.get("text", "")),
                    "key": str(action.get("key", "")),
                }
                parsed = self._normalize_action(parsed, observation)
                self._log_action(raw=raw, parsed=parsed, parse_ok=True, error=None)
                return parsed
            except (ValueError, TypeError):
                self._log_action(raw=raw, parsed=None, parse_ok=False, error="invalid_json_or_schema")
                continue
        fallback = {"action_type": "wait", "element_id": 0, "text": "", "key": ""}
        self._log_action(raw="", parsed=fallback, parse_ok=False, error="fallback_wait_action")
        return fallback

    def _generate_raw_action(self, observation: dict[str, Any], user_prompt: str) -> str:
        if self.cfg.use_real_llm:
            try:
                raw = self._query_backend(user_prompt)
                if raw.strip():
                    return raw
            except Exception as exc:  # noqa: BLE001
                self._log_action(raw="", parsed=None, parse_ok=False, error=f"real_llm_error:{exc}")
        return self._simulate_model_output(observation)

    def _query_backend(self, user_prompt: str) -> str:
        backend = self.cfg.backend.strip().lower()
        if backend == "ollama":
            return self._query_ollama(user_prompt)
        raise ValueError(f"unsupported llm backend: {self.cfg.backend}")

    def _query_ollama(self, user_prompt: str) -> str:
        payload = {
            "model": self.cfg.model_name,
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature,
                "top_p": self.cfg.top_p,
                "num_predict": self.cfg.max_tokens,
            },
            "messages": [
                {"role": "system", "content": PROMPT_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        }
        base = self.cfg.endpoint.rstrip("/")
        chat_url = base + "/api/chat"
        resp = requests.post(chat_url, json=payload, timeout=self.cfg.timeout_seconds)
        if resp.status_code == 404:
            # Backward-compatible fallback for servers exposing generate endpoint only.
            generate_payload = {
                "model": self.cfg.model_name,
                "stream": False,
                "options": payload["options"],
                "prompt": f"{PROMPT_SYSTEM}\n\n{user_prompt}",
            }
            gen = requests.post(base + "/api/generate", json=generate_payload, timeout=self.cfg.timeout_seconds)
            gen.raise_for_status()
            gen_data = gen.json()
            return str(gen_data.get("response", "")).strip()
        resp.raise_for_status()
        data = resp.json()
        message = data.get("message", {})
        content = str(message.get("content", "")).strip()
        return content

    def _simulate_model_output(self, observation: dict[str, Any]) -> str:
        elements = observation.get("elements", [])
        query = extract_search_query(str(observation.get("instruction", "")))
        if not elements:
            return json.dumps(
                {"action_type": "wait", "element_id": 0, "text": "", "key": ""},
                ensure_ascii=True,
            )

        # Heuristic behavior for real browser environment:
        # 1) Prefer typing into a likely search input.
        # 2) Else click a likely product/link/button.
        search_candidate = None
        click_candidate = None
        fallback_click = None
        def _sim_skip(el: dict[str, Any]) -> bool:
            cls = str(el.get("class_name", "")).lower()
            t = str(el.get("text", "")).lower()
            href = str(el.get("href", "")).lower()
            if "skip" in cls or "skip to" in t:
                return True
            if href.startswith("#") and len(href) > 2 and any(x in href for x in ("content", "main", "footer")):
                return True
            return False

        for el in elements:
            text = str(el.get("text", "")).lower()
            role = str(el.get("role", "")).lower()
            tag = str(el.get("tag", "")).lower()
            is_text_input = bool(el.get("is_text_input", False))
            visible = bool(el.get("visible", True))
            enabled = bool(el.get("enabled", True))
            if not visible or not enabled or _sim_skip(el):
                continue
            if fallback_click is None and (tag in {"a", "button", "input"} or "link" in role or "button" in role):
                fallback_click = el
            if search_candidate is None and is_text_input and (
                "search" in text or "search" in str(el.get("input_type", "")).lower()
            ):
                search_candidate = el
            if click_candidate is None and (
                "add to cart" in text
                or "view" in text
                or "details" in text
                or "product" in text
                or "link" in role
                or "button" in role
            ):
                click_candidate = el

        if search_candidate is not None and self.rng.random() < 0.6:
            return json.dumps(
                {
                    "action_type": "type",
                    "element_id": int(search_candidate["id"]),
                    "text": query,
                    "key": "",
                },
                ensure_ascii=True,
            )

        target = click_candidate if click_candidate is not None else fallback_click
        if target is None:
            return json.dumps(
                {"action_type": "wait", "element_id": 0, "text": "", "key": ""},
                ensure_ascii=True,
            )
        return json.dumps(
            {"action_type": "click", "element_id": int(target["id"]), "text": "", "key": ""},
            ensure_ascii=True,
        )

    def _log_action(
        self,
        raw: str,
        parsed: dict[str, Any] | None,
        parse_ok: bool,
        error: str | None,
    ) -> None:
        payload = {
            "raw_output": raw,
            "parse_ok": parse_ok,
            "parsed_action": parsed,
            "error": error,
        }
        if self.debug:
            print(f"[llm-policy] {json.dumps(payload, ensure_ascii=True)}", flush=True)
        if self.trace_path is not None:
            with open(self.trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _normalize_action(self, action: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
        action_type = str(action.get("action_type", "wait")).strip().lower()
        element_id = int(action.get("element_id", 0))
        text = str(action.get("text", ""))
        key = str(action.get("key", ""))
        elements = observation.get("elements", [])

        def _element_skip_like(el: dict[str, Any]) -> bool:
            cls = str(el.get("class_name", "")).lower()
            t = str(el.get("text", "")).lower().strip()
            href = str(el.get("href", "")).strip().lower()
            if "skip" in cls:
                return True
            if "skip to" in t:
                return True
            if href.startswith("#") and len(href) > 2 and any(x in href for x in ("content", "main", "footer", "nav")):
                return True
            return False

        def _first_text_input_id() -> int:
            for el in elements:
                if bool(el.get("is_text_input", False)) and bool(el.get("visible", True)) and bool(el.get("enabled", True)):
                    return int(el.get("id", 0))
            return 0

        def _first_clickable_id() -> int:
            for el in elements:
                if _element_skip_like(el):
                    continue
                role = str(el.get("role", "")).lower()
                tag = str(el.get("tag", "")).lower()
                if (
                    bool(el.get("visible", True))
                    and bool(el.get("enabled", True))
                    and (tag in {"a", "button", "input"} or "button" in role or "link" in role)
                ):
                    return int(el.get("id", 0))
            return 0

        if action_type in {"search", "input", "enter_text", "fill"}:
            query = extract_search_query(str(observation.get("instruction", "")))
            return {
                "action_type": "type",
                "element_id": _first_text_input_id(),
                "text": text if text else query,
                "key": "",
            }
        if action_type in {"submit", "enter"}:
            return {
                "action_type": "press",
                "element_id": 0,
                "text": "",
                "key": key if key else "Enter",
            }
        if action_type in {"tap", "select"}:
            return {
                "action_type": "click",
                "element_id": element_id if element_id > 0 else _first_clickable_id(),
                "text": "",
                "key": "",
            }
        if action_type in {"back"}:
            return {"action_type": "go_back", "element_id": 0, "text": "", "key": ""}
        if action_type not in {"click", "type", "press", "scroll", "go_back", "wait", "stop"}:
            return {"action_type": "wait", "element_id": 0, "text": "", "key": ""}
        if action_type == "type":
            id_set = {int(el.get("id", 0)) for el in elements if bool(el.get("is_text_input", False))}
            if not id_set:
                return {"action_type": "wait", "element_id": 0, "text": "", "key": ""}
            if element_id not in id_set:
                element_id = _first_text_input_id()
            if element_id <= 0:
                return {"action_type": "wait", "element_id": 0, "text": "", "key": ""}
        if action_type == "click":
            clickable_ids: set[int] = set()
            for el in elements:
                if _element_skip_like(el):
                    continue
                role = str(el.get("role", "")).lower()
                tag = str(el.get("tag", "")).lower()
                if (
                    bool(el.get("visible", True))
                    and bool(el.get("enabled", True))
                    and (tag in {"a", "button", "input"} or "button" in role or "link" in role)
                ):
                    clickable_ids.add(int(el.get("id", 0)))
            if not clickable_ids:
                return {"action_type": "wait", "element_id": 0, "text": "", "key": ""}
            if element_id not in clickable_ids:
                element_id = _first_clickable_id()
            if element_id <= 0:
                return {"action_type": "wait", "element_id": 0, "text": "", "key": ""}
        return {
            "action_type": action_type,
            "element_id": element_id,
            "text": text,
            "key": key,
        }

