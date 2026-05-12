from __future__ import annotations

from typing import Any

from .task_utils import parse_task_spec


class ScriptedShoppingPolicy:
    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        spec = parse_task_spec(str(obs.get("instruction", "")))
        elements = list(obs.get("elements", []))
        page = str(obs.get("page", "")).lower()

        text_inputs = [e for e in elements if bool(e.get("is_text_input", False)) and _is_usable(e)]
        product_links = [e for e in elements if _is_productish_click(e)]
        cart_links = [e for e in elements if _looks_like_cart_link(e)]
        checkout_links = [e for e in elements if _looks_like_checkout_link(e)]
        add_to_cart_buttons = [e for e in elements if _looks_like_add_to_cart(e)]

        query = spec.product_query or "laptop"
        if text_inputs and not _query_already_reflected(obs, query):
            return {
                "action_type": "type",
                "element_id": int(text_inputs[0]["id"]),
                "text": query,
                "key": "",
            }

        if "product" in page or "catalog/product" in page:
            if spec.require_cart_or_checkout and add_to_cart_buttons:
                return _click(add_to_cart_buttons[0])
            return _stop()

        if spec.require_cart_or_checkout and "/checkout/cart" in page:
            if "checkout" in page:
                return _stop()
            if checkout_links:
                return _click(checkout_links[0])
            return _stop()

        if spec.require_cart_or_checkout and ("/checkout" in page or "onepage" in page):
            return _stop()

        keyword_ranked = sorted(product_links, key=lambda e: _keyword_overlap_score(e, spec.product_keywords), reverse=True)
        for candidate in keyword_ranked:
            if _keyword_overlap_score(candidate, spec.product_keywords) <= 0:
                continue
            return _click(candidate)

        if spec.require_cart_or_checkout and cart_links:
            return _click(cart_links[0])

        if keyword_ranked:
            return _click(keyword_ranked[0])
        if text_inputs:
            return {
                "action_type": "type",
                "element_id": int(text_inputs[0]["id"]),
                "text": query,
                "key": "",
            }
        return {"action_type": "wait", "element_id": 0, "text": "", "key": ""}


def _click(el: dict[str, Any]) -> dict[str, Any]:
    return {"action_type": "click", "element_id": int(el["id"]), "text": "", "key": ""}


def _stop() -> dict[str, Any]:
    return {"action_type": "stop", "element_id": 0, "text": "", "key": ""}


def _is_usable(el: dict[str, Any]) -> bool:
    if not bool(el.get("visible", False)) or not bool(el.get("enabled", False)):
        return False
    if _looks_like_chrome(el):
        return False
    return True


def _looks_like_chrome(el: dict[str, Any]) -> bool:
    text = str(el.get("text", "")).lower()
    href = str(el.get("href", "")).lower().strip()
    cls = str(el.get("class_name", "")).lower()
    block = ("logo", "account", "wishlist", "compare", "advanced search", "skip to", "my account")
    if any(token in text for token in block):
        return True
    if "skip" in cls:
        return True
    if href.startswith("#"):
        return True
    return False


def _is_productish_click(el: dict[str, Any]) -> bool:
    if not _is_usable(el):
        return False
    role = str(el.get("role", "")).lower()
    tag = str(el.get("tag", "")).lower()
    text = str(el.get("text", "")).lower()
    return (tag in {"a", "button"} or "link" in role or "button" in role) and "cart" not in text and "checkout" not in text


def _looks_like_cart_link(el: dict[str, Any]) -> bool:
    if not _is_usable(el):
        return False
    txt = str(el.get("text", "")).lower()
    href = str(el.get("href", "")).lower()
    return "cart" in txt or "/checkout/cart" in href


def _looks_like_checkout_link(el: dict[str, Any]) -> bool:
    if not _is_usable(el):
        return False
    txt = str(el.get("text", "")).lower()
    href = str(el.get("href", "")).lower()
    return "checkout" in txt or "/checkout" in href


def _looks_like_add_to_cart(el: dict[str, Any]) -> bool:
    if not _is_usable(el):
        return False
    txt = str(el.get("text", "")).lower()
    return "add to cart" in txt


def _query_already_reflected(obs: dict[str, Any], query: str) -> bool:
    q = query.lower().strip()
    if not q:
        return False
    page = str(obs.get("page", "")).lower()
    if q.replace(" ", "+") in page or q.replace(" ", "%20") in page:
        return True
    for el in obs.get("elements", []):
        if not bool(el.get("is_text_input", False)):
            continue
        text = str(el.get("text", "")).lower()
        if q in text:
            return True
    return False


def _keyword_overlap_score(el: dict[str, Any], keywords: list[str]) -> int:
    text = str(el.get("text", "")).lower()
    href = str(el.get("href", "")).lower()
    score = 0
    for keyword in keywords:
        k = keyword.lower()
        if k and (k in text or k in href):
            score += 2 if " " in k else 1
    return score
