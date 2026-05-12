from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from .task_utils import TaskSpec, extract_search_query, parse_task_spec


@dataclass
class StepResult:
    observation: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]


class WebArenaShoppingEnv:
    """Browser-driven WebArena shopping environment wrapper."""

    def __init__(
        self,
        shopping_url: str,
        admin_url: str,
        max_steps: int,
        seed: int,
        headless: bool = True,
        debug: bool = False,
        trace_path: str | None = None,
        repair_invalid_actions: bool = False,
        clear_state_on_reset: bool = True,
        force_click: bool = False,
        reset_url: str | None = None,
    ) -> None:
        self.shopping_url = shopping_url
        self.admin_url = admin_url
        self.max_steps = max_steps
        self.seed = seed
        self.headless = headless
        self.debug = debug
        self.trace_path = Path(trace_path) if trace_path else None
        self.repair_invalid_actions = repair_invalid_actions
        self.clear_state_on_reset = clear_state_on_reset
        self.force_click = force_click
        self.reset_url = reset_url
        self._step = 0
        self._history: list[dict[str, Any]] = []
        self._instruction = ""
        self._task_spec = TaskSpec(
            raw_instruction="",
            product_query="",
            product_keywords=[],
            max_price=None,
            require_cart_or_checkout=False,
        )
        self._id_to_selector: dict[int, str] = {}
        self._id_to_meta: dict[int, dict[str, Any]] = {}
        self._play = None
        self._browser = None
        self._context = None
        self._page = None
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)

    def health_check(self) -> None:
        for url in (self.shopping_url, self.admin_url):
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()

    def reset(self, task: str) -> dict[str, Any]:
        self._step = 0
        self._history = []
        self._instruction = task
        self._task_spec = parse_task_spec(task)
        self._ensure_browser()
        assert self._page is not None
        if self.clear_state_on_reset:
            self._clear_runtime_state()
        self._page.goto(self.shopping_url, wait_until="domcontentloaded", timeout=30000)
        return self._build_observation()

    def step(self, action: dict[str, Any]) -> StepResult:
        self._step += 1
        done = self._step >= self.max_steps
        shaped_reward = -0.01
        terminal_reward = 0.0
        assert self._page is not None
        info: dict[str, Any] = {"step": self._step, "action_ok": True, "verified_success": False}
        url_before = self._page.url

        try:
            valid, reason, normalized_action = self._validate_action(action)
            action = normalized_action
            action_type = str(action.get("action_type", "wait"))
            if not valid:
                info["action_ok"] = False
                info["error"] = reason
                shaped_reward = -0.10
                self._history.append(action)
                obs = self._build_observation()
                info["shaped_reward"] = shaped_reward
                info["terminal_reward"] = terminal_reward
                self._log_step(
                    {
                        "step": self._step,
                        "action": action,
                        "url_before": url_before,
                        "url_after": self._page.url,
                        "action_ok": False,
                        "error": reason,
                        "reward": shaped_reward + terminal_reward,
                        "done": done,
                        "verifier": None,
                    }
                )
                return StepResult(observation=obs, reward=shaped_reward + terminal_reward, done=done, info=info)
            if action_type == "click":
                selector = self._selector_for_action(action, preferred="click")
                self._page.locator(selector).first.click(timeout=4000, force=self.force_click)
                self._safe_wait_nav()
            elif action_type == "type":
                selector = self._selector_for_action(action, preferred="type")
                text = self._normalize_type_text(str(action.get("text", "")))
                field = self._page.locator(selector).first
                field.click(timeout=2000)
                field.fill(text, timeout=4000)
                field.press("Enter", timeout=4000)
                self._safe_wait_nav()
            elif action_type == "press":
                key = str(action.get("key", "Enter")) or "Enter"
                self._page.keyboard.press(key)
            elif action_type == "scroll":
                self._page.mouse.wheel(0, 500)
            elif action_type == "go_back":
                self._page.go_back(timeout=4000)
            elif action_type == "stop":
                done = True
                verifier = self._compute_terminal_verdict()
                terminal_reward = 1.0 if verifier["verified_success"] else 0.0
                info["verifier"] = verifier
                info["verified_success"] = bool(verifier.get("verified_success", False))
            else:
                self._page.wait_for_timeout(300)
        except Exception as exc:  # noqa: BLE001
            info["action_ok"] = False
            info["error"] = str(exc)
            shaped_reward = -0.10

        self._history.append(action)
        obs = self._build_observation()
        shaping = self._compute_shaping(action=action, observation=obs, action_ok=bool(info.get("action_ok", True)))
        shaped_reward += shaping
        if "verifier" not in info:
            verifier = self._compute_terminal_verdict() if done else None
            info["verifier"] = verifier
            info["verified_success"] = bool(verifier and verifier.get("verified_success", False))
        info["shaped_reward"] = shaped_reward
        info["terminal_reward"] = terminal_reward
        reward = shaped_reward + terminal_reward
        self._log_step(
            {
                "step": self._step,
                "action": action,
                "url_before": url_before,
                "url_after": self._page.url,
                "action_ok": info.get("action_ok", True),
                "error": info.get("error"),
                "reward": reward,
                "done": done,
                "verifier": info.get("verifier"),
            }
        )
        return StepResult(observation=obs, reward=reward, done=done, info=info)

    def _safe_wait_nav(self) -> None:
        assert self._page is not None
        try:
            self._page.wait_for_load_state("domcontentloaded", timeout=1200)
        except Exception:  # noqa: BLE001
            pass
        try:
            self._page.wait_for_load_state("networkidle", timeout=1200)
        except Exception:  # noqa: BLE001
            pass

    def _clear_runtime_state(self) -> None:
        assert self._context is not None
        assert self._page is not None
        if self.reset_url:
            try:
                requests.post(self.reset_url, timeout=5)
            except Exception:  # noqa: BLE001
                pass
        try:
            self._context.clear_cookies()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._page.goto(self.shopping_url, wait_until="domcontentloaded", timeout=15000)
        except Exception:  # noqa: BLE001
            pass
        try:
            self._page.evaluate(
                """
                () => {
                  try { window.localStorage && window.localStorage.clear(); } catch (e) {}
                  try { window.sessionStorage && window.sessionStorage.clear(); } catch (e) {}
                  return true;
                }
                """
            )
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        if self._context is not None:
            self._context.close()
        if self._browser is not None:
            self._browser.close()
        if self._play is not None:
            self._play.stop()
        self._play = None
        self._browser = None
        self._context = None
        self._page = None

    def _ensure_browser(self) -> None:
        if self._page is not None:
            return
        self._play = sync_playwright().start()
        self._browser = self._play.chromium.launch(headless=self.headless)
        self._context = self._browser.new_context(viewport={"width": 1280, "height": 720})
        self._page = self._context.new_page()

    def _build_observation(self) -> dict[str, Any]:
        assert self._page is not None
        self._page.wait_for_timeout(200)
        elements = self._extract_elements()
        return {
            "instruction": self._instruction,
            "page": self._page.url,
            "elements": elements,
            "history": self._history[-10:],
        }

    def _extract_elements(self) -> list[dict[str, Any]]:
        assert self._page is not None
        try:
            rows = self._page.evaluate(
                """
                () => {
                  function cssPath(el) {
                    if (!el || !el.parentElement) {
                      return "";
                    }
                    const parts = [];
                    let cur = el;
                    let depth = 0;
                    while (cur && cur.nodeType === 1 && depth < 4) {
                      const tag = cur.tagName.toLowerCase();
                      if (cur.id) {
                        parts.unshift("#" + CSS.escape(cur.id));
                        break;
                      }
                      let idx = 1;
                      let sib = cur;
                      while ((sib = sib.previousElementSibling) != null) {
                        if (sib.tagName === cur.tagName) idx += 1;
                      }
                      parts.unshift(tag + ":nth-of-type(" + idx + ")");
                      cur = cur.parentElement;
                      depth += 1;
                    }
                    return parts.join(" > ");
                  }

                  function isVisible(el) {
                    const style = window.getComputedStyle(el);
                    if (style.display === "none" || style.visibility === "hidden" || style.opacity === "0") {
                      return false;
                    }
                    const rect = el.getBoundingClientRect();
                    return rect.width > 0 && rect.height > 0;
                  }

                  function isSkipOrChromeLink(el) {
                    const tag = (el.tagName || "").toLowerCase();
                    if (tag !== "a" && tag !== "button") {
                      return false;
                    }
                    const cls = String(el.className || "").toLowerCase();
                    const href = (el.getAttribute("href") || "").trim();
                    const txt = ((el.innerText || el.textContent || "")).trim().toLowerCase();
                    if (cls.includes("skip") || cls.includes("skiptocontent")) {
                      return true;
                    }
                    if (txt.includes("skip to content") || txt.includes("skip to main")) {
                      return true;
                    }
                    if (href.startsWith("#") && href.length > 2) {
                      const h = href.toLowerCase();
                      if (h.includes("content") || h.includes("main") || h.includes("footer") || h.includes("navigation")) {
                        return true;
                      }
                    }
                    return false;
                  }

                  const selectors = [
                    "a[href]",
                    "button",
                    "input",
                    "select",
                    "textarea",
                    "[role='button']",
                    "[role='link']"
                  ];
                  const nodes = Array.from(document.querySelectorAll(selectors.join(",")))
                    .filter((el) => !isSkipOrChromeLink(el))
                    .slice(0, 160);
                  return nodes.map((el, idx) => {
                    const text = (el.innerText || el.value || el.getAttribute("aria-label") || "").toString().slice(0, 120);
                    const href = (el.getAttribute("href") || "").toString();
                    const class_name = String(el.className || "");
                    const tag = (el.tagName || "").toLowerCase();
                    const role = (el.getAttribute("role") || "").toLowerCase();
                    const inputType = (el.getAttribute("type") || "").toLowerCase();
                    const visible = isVisible(el);
                    const disabled = !!el.disabled || el.getAttribute("aria-disabled") === "true";
                    const isTextInput = (tag === "input" && !["checkbox","radio","submit","button","hidden","file"].includes(inputType)) || tag === "textarea";
                    let selector = "";
                    if (el.id) {
                      selector = "#" + CSS.escape(el.id);
                    } else if (el.getAttribute("name")) {
                      selector = el.tagName.toLowerCase() + "[name='" + el.getAttribute("name").replace(/'/g, "\\\\'") + "']";
                    } else if (el.getAttribute("aria-label")) {
                      selector = el.tagName.toLowerCase() + "[aria-label='" + el.getAttribute("aria-label").replace(/'/g, "\\\\'") + "']";
                    } else if (el.getAttribute("placeholder")) {
                      selector = el.tagName.toLowerCase() + "[placeholder='" + el.getAttribute("placeholder").replace(/'/g, "\\\\'") + "']";
                    } else {
                      selector = cssPath(el);
                    }
                    return {
                      role: role || tag,
                      tag: tag,
                      input_type: inputType,
                      visible: visible,
                      enabled: !disabled,
                      is_text_input: isTextInput,
                      text: text,
                      href: href,
                      class_name: class_name,
                      selector: selector
                    };
                  });
                }
                """
            )
        except PlaywrightTimeoutError:
            rows = []
        except Exception:  # noqa: BLE001
            # Navigation can invalidate JS execution context mid-evaluation.
            # Return empty elements for this step and recover next step.
            rows = []
        self._id_to_selector = {}
        self._id_to_meta = {}
        elements: list[dict[str, Any]] = []
        for idx, row in enumerate(rows, start=1):
            selector = str(row.get("selector", "")).strip()
            visible = bool(row.get("visible", False))
            enabled = bool(row.get("enabled", False))
            if not selector or not visible or not enabled:
                continue
            self._id_to_selector[idx] = selector
            self._id_to_meta[idx] = {
                "role": str(row.get("role", "")),
                "tag": str(row.get("tag", "")),
                "input_type": str(row.get("input_type", "")),
                "is_text_input": bool(row.get("is_text_input", False)),
                "visible": visible,
                "enabled": enabled,
                "text": str(row.get("text", "")),
                "href": str(row.get("href", "")),
                "class_name": str(row.get("class_name", "")),
            }
            elements.append(
                {
                    "id": idx,
                    "role": str(row.get("role", "")),
                    "tag": str(row.get("tag", "")),
                    "input_type": str(row.get("input_type", "")),
                    "is_text_input": bool(row.get("is_text_input", False)),
                    "visible": visible,
                    "enabled": enabled,
                    "text": str(row.get("text", "")),
                    "href": str(row.get("href", "")),
                    "class_name": str(row.get("class_name", "")),
                }
            )
        return elements[:50]

    def _bad_click_meta(self, meta: dict[str, Any]) -> bool:
        """Skip navigation chrome that causes timeouts (skip links, in-page anchors)."""
        cls = str(meta.get("class_name", "")).lower()
        text = str(meta.get("text", "")).lower().strip()
        href = str(meta.get("href", "")).strip().lower()
        if "skip" in cls:
            return True
        if "skip to" in text:
            return True
        if href.startswith("#") and len(href) > 2:
            if any(x in href for x in ("content", "main", "footer", "nav", "navigation")):
                return True
        return False

    def _css_fallback_chain(self, preferred: str) -> list[str]:
        """Last-resort selectors when ID maps are empty (stale obs / torn navigation)."""
        if preferred == "type":
            return [
                "#search",
                "input[name='q']",
                "input[type='search']",
                "input[type='text']",
                "textarea",
            ]
        return [
            "main a.product-item-link",
            "ol.products li a",
            "a.product-item-link",
            "[data-role='priceBox'] a",
            "button[type='submit']",
            "form button.search",
        ]

    def _selector_for_action(self, action: dict[str, Any], preferred: str = "click") -> str:
        element_id = int(action.get("element_id", 0))
        selector = self._id_to_selector.get(element_id)
        if selector:
            return selector
        fallback_id = self._fallback_element_id(preferred)
        selector = self._id_to_selector.get(fallback_id)
        if selector:
            return selector
        assert self._page is not None
        for css in self._css_fallback_chain(preferred):
            loc = self._page.locator(css)
            try:
                if loc.count() > 0:
                    return css
            except Exception:  # noqa: BLE001
                continue
        raise ValueError(f"unknown element_id={element_id} and no fallback selector")

    def _fallback_element_id(self, preferred: str) -> int:
        # Prefer stable, interactable controls; avoid hard-failing on stale IDs after navigation.
        if preferred == "type":
            for id_, meta in sorted(self._id_to_meta.items()):
                if bool(meta.get("is_text_input", False)) and bool(meta.get("visible", False)) and bool(meta.get("enabled", False)):
                    return id_
        for id_, meta in sorted(self._id_to_meta.items()):
            role = str(meta.get("role", "")).lower()
            tag = str(meta.get("tag", "")).lower()
            if self._bad_click_meta(meta):
                continue
            if bool(meta.get("visible", False)) and bool(meta.get("enabled", False)) and (
                tag in {"a", "button", "input"} or "button" in role or "link" in role
            ):
                return id_
        return 0

    def _compute_terminal_reward(self) -> float:
        # Legacy helper retained for compatibility. Prefer verifier-driven success.
        verdict = self._compute_terminal_verdict()
        return 1.0 if verdict["verified_success"] else 0.0

    def _compute_terminal_verdict(self) -> dict[str, Any]:
        assert self._page is not None
        visible_titles = self._visible_product_titles()
        observed_prices = self._extract_visible_prices_from_dom()
        matched_keywords = [k for k in self._task_spec.product_keywords if any(k in title for title in visible_titles)]
        keyword_ok = True if not self._task_spec.product_keywords else bool(matched_keywords)
        price_ok = True
        if self._task_spec.max_price is not None:
            price_ok = any(p <= self._task_spec.max_price for p in observed_prices)
        cart_ok = self._cart_or_checkout_ok() if self._task_spec.require_cart_or_checkout else True

        # Ensure terminal success is not triggered by unrelated landing pages.
        interaction_ok = len(self._history) >= 2
        verified = keyword_ok and price_ok and cart_ok and interaction_ok
        return {
            "verified_success": verified,
            "keyword_ok": keyword_ok,
            "price_ok": price_ok,
            "cart_or_checkout_ok": cart_ok,
            "interaction_ok": interaction_ok,
            "matched_keywords": matched_keywords,
            "max_price": self._task_spec.max_price,
            "observed_prices_sample": observed_prices[:10],
            "task_spec": {
                "raw_instruction": self._task_spec.raw_instruction,
                "product_keywords": self._task_spec.product_keywords,
                "require_cart_or_checkout": self._task_spec.require_cart_or_checkout,
            },
        }

    def _safe_body_text(self) -> str:
        assert self._page is not None
        try:
            text = self._page.locator("body").inner_text(timeout=3000)
        except Exception:  # noqa: BLE001
            text = ""
        return text.lower()

    def _visible_product_titles(self) -> list[str]:
        assert self._page is not None
        selectors = ".product-item-name, .product-item-link, .page-title, .product-info-main"
        try:
            rows = self._page.evaluate(
                """
                (sel) => Array.from(document.querySelectorAll(sel))
                  .map((el) => ((el.innerText || el.textContent || '').trim().toLowerCase()))
                  .filter((t) => !!t)
                  .slice(0, 60)
                """,
                selectors,
            )
        except Exception:  # noqa: BLE001
            rows = []
        return [str(x) for x in rows]

    def _extract_visible_prices_from_dom(self) -> list[float]:
        assert self._page is not None
        try:
            values = self._page.evaluate(
                """
                () => {
                  const nodes = Array.from(document.querySelectorAll(".price, [data-price-amount]"));
                  const out = [];
                  for (const n of nodes) {
                    const dataAmount = n.getAttribute("data-price-amount");
                    if (dataAmount) {
                      const v = parseFloat(dataAmount);
                      if (!Number.isNaN(v)) { out.push(v); continue; }
                    }
                    const txt = ((n.innerText || n.textContent || "")).trim();
                    const m = txt.match(/(\\d+(?:\\.\\d{1,2})?)/);
                    if (m) {
                      const v = parseFloat(m[1]);
                      if (!Number.isNaN(v)) { out.push(v); }
                    }
                  }
                  return out.slice(0, 100);
                }
                """
            )
        except Exception:  # noqa: BLE001
            values = []
        out: list[float] = []
        for v in values:
            try:
                out.append(float(v))
            except Exception:  # noqa: BLE001
                continue
        return out

    def _cart_or_checkout_ok(self) -> bool:
        assert self._page is not None
        path = urlparse(self._page.url).path.lower()
        if path.startswith("/checkout/cart") or path.startswith("/checkout/onepage") or path.startswith("/checkout"):
            return True
        selectors = (".cart.item", ".minicart-items .product-item", "#shopping-cart-table tbody tr")
        for selector in selectors:
            try:
                if self._page.locator(selector).count() > 0:
                    return True
            except Exception:  # noqa: BLE001
                continue
        return False

    def _compute_shaping(self, action: dict[str, Any], observation: dict[str, Any], action_ok: bool) -> float:
        assert self._page is not None
        if not action_ok:
            return 0.0
        shaped = 0.0
        query = extract_search_query(self._instruction).lower()
        page = str(observation.get("page", "")).lower()
        if "search" in page or (query and (query.replace(" ", "+") in page or query.replace(" ", "%20") in page)):
            shaped += 0.05
        if self._task_spec.product_keywords:
            titles = self._visible_product_titles()
            if any(keyword in " ".join(titles) for keyword in self._task_spec.product_keywords):
                shaped += 0.10
        if ("catalog/product" in page or "product" in page) and self._compute_terminal_verdict().get("keyword_ok", False):
            if self._compute_terminal_verdict().get("price_ok", True):
                shaped += 0.25
        if self._task_spec.require_cart_or_checkout and self._cart_or_checkout_ok():
            shaped += 0.50
        if str(action.get("action_type", "")).lower() == "stop":
            verdict = self._compute_terminal_verdict()
            if verdict.get("verified_success", False):
                shaped += 1.0
        return shaped

    def _validate_action(self, action: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        action_type = str(action.get("action_type", "wait")).strip().lower()
        normalized = {
            "action_type": action_type,
            "element_id": action.get("element_id", 0),
            "text": str(action.get("text", "")),
            "key": str(action.get("key", "")),
        }
        allowed = {"click", "type", "press", "scroll", "go_back", "wait", "stop"}
        if action_type not in allowed:
            return False, f"invalid action_type={action_type}", normalized
        try:
            normalized["element_id"] = int(normalized["element_id"])
        except Exception:  # noqa: BLE001
            return False, "element_id is not an integer", normalized
        if action_type in {"click", "type"} and normalized["element_id"] <= 0:
            return False, "element_id must be > 0 for click/type", normalized
        if action_type == "type":
            meta = self._id_to_meta.get(normalized["element_id"], {})
            if normalized["element_id"] in self._id_to_meta and not bool(meta.get("is_text_input", False)):
                if self.repair_invalid_actions:
                    normalized["element_id"] = self._fallback_element_id("type")
                else:
                    return False, "type target is not a text input and repair is disabled", normalized
            if normalized["element_id"] <= 0:
                return False, "type action requires a text-input element", normalized
        if action_type == "click":
            meta = self._id_to_meta.get(normalized["element_id"], {})
            click_ok = (
                str(meta.get("tag", "")) in {"a", "button", "input"}
                or "button" in str(meta.get("role", ""))
                or "link" in str(meta.get("role", ""))
            )
            if normalized["element_id"] in self._id_to_meta and (not click_ok or self._bad_click_meta(meta)):
                if self.repair_invalid_actions:
                    normalized["element_id"] = self._fallback_element_id("click")
                else:
                    return False, "click target is not valid and repair is disabled", normalized
            if normalized["element_id"] <= 0:
                return False, "click action target is not clickable", normalized
        return True, "", normalized

    def _normalize_type_text(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return extract_search_query(self._instruction)
        # Prevent runaway concatenation from repeated self-generated search strings.
        words = text.split()
        if len(words) > 8:
            text = " ".join(words[:8])
        return text

    def _log_step(self, event: dict[str, Any]) -> None:
        if self.debug:
            print(f"[env-step] {json.dumps(event, ensure_ascii=True)}", flush=True)
        if self.trace_path is None:
            return
        with open(self.trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=True) + "\n")

