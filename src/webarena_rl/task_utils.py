from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class TaskSpec:
    raw_instruction: str
    product_query: str
    product_keywords: list[str]
    max_price: float | None
    require_cart_or_checkout: bool


_STOP_PATTERNS = (
    r"\bunders?\b",
    r"\bbelow\b",
    r"\bless than\b",
    r"\band add\b",
    r"\band open\b",
    r"\bopen details\b",
    r"\bopen product\b",
    r"\bnavigate\b",
    r"\bto cart\b",
    r"\bto checkout\b",
    r"\bcheckout\b",
)


def _clean_text(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"[^\w\s$.-]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def parse_task_spec(task: str) -> TaskSpec:
    clean_task = _clean_text(task)
    max_price = _extract_max_price(clean_task)
    product_query = _extract_product_phrase(clean_task)
    require_cart = _requires_cart_or_checkout(clean_task)
    keywords = _build_keywords(product_query)
    return TaskSpec(
        raw_instruction=task,
        product_query=product_query,
        product_keywords=keywords,
        max_price=max_price,
        require_cart_or_checkout=require_cart,
    )


def extract_search_query(task: str) -> str:
    spec = parse_task_spec(task)
    if spec.product_query:
        return spec.product_query
    return "laptop"


def _extract_max_price(clean_task: str) -> float | None:
    match = re.search(r"(?:under|below|less than)\s*\$?\s*(\d+(?:\.\d{1,2})?)", clean_task)
    if not match:
        return None
    return float(match.group(1))


def _extract_product_phrase(clean_task: str) -> str:
    patterns = (
        r"\bsearch for\s+(.+)$",
        r"\blook for\s+(.+)$",
        r"\bfind\s+(.+)$",
        r"\badd\s+(.+)$",
    )
    phrase = ""
    for pattern in patterns:
        match = re.search(pattern, clean_task)
        if match:
            phrase = match.group(1).strip()
            break
    if not phrase:
        return ""
    for stop in _STOP_PATTERNS:
        phrase = re.split(stop, phrase, maxsplit=1)[0].strip()
    phrase = re.sub(r"^(a|an|the)\s+", "", phrase).strip()
    return phrase


def _build_keywords(product_query: str) -> list[str]:
    if not product_query:
        return []
    words = [w for w in product_query.split() if len(w) >= 3]
    result: list[str] = [product_query]
    for word in words:
        if word not in result:
            result.append(word)
    return result


def _requires_cart_or_checkout(clean_task: str) -> bool:
    markers = ("add to cart", "navigate to cart", "cart", "checkout")
    return any(marker in clean_task for marker in markers)
