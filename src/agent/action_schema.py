def execute_action(page, action):
    action_type = action.get("type")

    if action_type == "TYPE":
        selector = action["target"]
        value = action["value"]
        page.fill(selector, value)

    elif action_type == "CLICK":
        selector = action["target"]
        page.click(selector)

    elif action_type == "TYPE_AND_CLICK_SEARCH":
        page.fill('input[name="q"]', action["value"])
        page.get_by_role("button", name="Search").click()

    elif action_type == "STOP":
        pass

    else:
        raise ValueError(f"Unsupported action type: {action_type}")