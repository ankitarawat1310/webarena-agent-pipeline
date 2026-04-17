from typing import Any, Dict, List
import re

SEARCH_TERMS = [
    "headphones",
    "backpack",
    "keyboard",
    "laptop",
    "bottle",
    "shirt",
    "watch",
    "shoes",
    "mouse",
    "phone",
    "bag",
]


def _find_search_box(observation: Dict[str, Any]) -> str:
    visible_inputs: List[Dict[str, Any]] = observation.get("visible_inputs", [])

    for inp in visible_inputs:
        name = str(inp.get("name", "")).strip().lower()
        placeholder = str(inp.get("placeholder", "")).strip().lower()

        if name == "q":
            return 'input[name="q"]'

        if "search" in placeholder:
            if inp.get("name"):
                return f'input[name="{inp.get("name")}"]'
            return 'input[placeholder*="Search"]'

    if observation.get("search_box_present", False):
        return 'input[name="q"]'

    return ""


def _get_search_box_value(observation: Dict[str, Any]) -> str:
    visible_inputs: List[Dict[str, Any]] = observation.get("visible_inputs", [])

    for inp in visible_inputs:
        name = str(inp.get("name", "")).strip().lower()
        if name == "q":
            return str(inp.get("value", "")).strip().lower()

    return ""


def _has_button(observation: Dict[str, Any], button_text: str) -> bool:
    buttons = [str(btn).strip().lower() for btn in observation.get("visible_buttons", [])]
    return button_text.strip().lower() in buttons


def _extract_search_term(instruction_lower: str) -> str:
    for term in SEARCH_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", instruction_lower):
            return term
    return ""


def choose_action(observation: Dict[str, Any], instruction: str) -> Dict[str, str]:
    instruction_lower = instruction.lower()

    search_box_selector = _find_search_box(observation)
    search_box_value = _get_search_box_value(observation)
    has_search_button = _has_button(observation, "Search")
    has_add_to_cart_button = _has_button(observation, "Add to Cart")

    chosen_term = _extract_search_term(instruction_lower)

    # ----------------------------
    # 1. ADD-TO-CART TASKS
    # ----------------------------
    current_url = observation.get("url", "").lower()

    if "add it to cart" in instruction_lower or "add them to cart" in instruction_lower or "add to cart" in instruction_lower:
        if not chosen_term:
            return {"type": "STOP", "target": "", "value": ""}

    current_url = observation.get("url", "").lower()

    # Step 3a: search results page -> click Add to Cart
    if "catalogsearch" in current_url and has_add_to_cart_button:
        return {
            "type": "CLICK",
            "target": 'button:has-text("Add to Cart")',
            "value": ""
        }

    # Step 3b: product page opened with options/cart flow -> click Add to Cart again
    if "options=cart" in current_url and has_add_to_cart_button:
        return {
            "type": "CLICK",
            "target": 'button:has-text("Add to Cart")',
            "value": ""
        }

    # Step 1: type product
    if search_box_selector and search_box_value != chosen_term:
        return {
            "type": "TYPE",
            "target": search_box_selector,
            "value": chosen_term
        }

    # Step 2: click search
    if has_search_button and "catalogsearch" not in current_url:
        return {
            "type": "CLICK",
            "target": 'button:has-text("Search")',
            "value": ""
        }

        return {"type": "STOP", "target": "", "value": ""}

    # ----------------------------
    # 2. SEARCH TASKS
    # ----------------------------
    if "search" in instruction_lower or "find" in instruction_lower or "look for" in instruction_lower:
        if not chosen_term:
            return {"type": "STOP", "target": "", "value": ""}

        # Step 1: type product
        if search_box_selector and search_box_value != chosen_term:
            return {
                "type": "TYPE",
                "target": search_box_selector,
                "value": chosen_term
            }

        # Step 2: click search
        if has_search_button and "catalogsearch" not in observation.get("url", "").lower():
            return {
                "type": "CLICK",
                "target": 'button:has-text("Search")',
                "value": ""
            }

        return {"type": "STOP", "target": "", "value": ""}

    # ----------------------------
    # 3. NAVIGATION TASKS
    # ----------------------------
    if "sign in" in instruction_lower:
        return {"type": "CLICK", "target": 'a:has-text("Sign In")', "value": ""}

    elif "home" in instruction_lower:
        return {"type": "CLICK", "target": 'a:has-text("Home")', "value": ""}

    elif "cart page" in instruction_lower or "go to the cart" in instruction_lower:
        return {"type": "CLICK", "target": 'a:has-text("Cart")', "value": ""}

    return {
        "type": "STOP",
        "target": "",
        "value": ""
    }