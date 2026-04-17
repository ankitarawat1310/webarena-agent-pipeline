from typing import Any, Dict


def execute_action(page, action: Dict[str, Any]) -> None:
    """
    Execute a high-level agent action on a Playwright page.

    Supported action schema:
    {
        "type": "CLICK | TYPE | SELECT | CHECK | UNCHECK | PRESS | WAIT | SCROLL | NAVIGATE | STOP",
        "target": "CSS selector or logical target name",
        "value": "optional value / key / url / direction"
    }
    """
    action_type = str(action.get("type", "")).strip().upper()
    target = str(action.get("target", "")).strip()
    value = action.get("value", "")

    if action_type == "CLICK":
        if not target:
            raise ValueError("CLICK action requires a target.")
        page.click(action["target"])

    elif action_type == "TYPE":
        if not target:
            raise ValueError("TYPE action requires a target.")
        page.fill(target, str(value))

    elif action_type == "SELECT":
        if not target:
            raise ValueError("SELECT action requires a target.")
        page.select_option(target, str(value))

    elif action_type == "CHECK":
        if not target:
            raise ValueError("CHECK action requires a target.")
        page.check(target)

    elif action_type == "UNCHECK":
        if not target:
            raise ValueError("UNCHECK action requires a target.")
        page.uncheck(target)

    elif action_type == "PRESS":
        key_target = target if target else "body"
        locator = page.locator(key_target).first
        locator.press(str(value))

    elif action_type == "WAIT":
        wait_ms = int(value) if str(value).strip() else 2000
        page.wait_for_timeout(wait_ms)

    elif action_type == "SCROLL":
        direction = str(value).strip().lower()
        if direction == "down":
            page.mouse.wheel(0, 1200)
        elif direction == "up":
            page.mouse.wheel(0, -1200)
        else:
            raise ValueError("SCROLL value must be 'up' or 'down'.")

    elif action_type == "NAVIGATE":
        url = str(value).strip() if str(value).strip() else target
        if not url:
            raise ValueError("NAVIGATE action requires a URL in target or value.")
        page.goto(url)

    elif action_type == "STOP":
        # No operation; used to indicate task completion or no valid action.
        pass

    else:
        raise ValueError(f"Unsupported action type: {action_type}")