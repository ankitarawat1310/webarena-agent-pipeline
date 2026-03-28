def build_input_text(step):
    instruction = step["instruction"]

    obs = step["observation_before"]

    buttons = obs.get("visible_buttons", [])
    inputs = obs.get("visible_inputs", [])

    input_fields = [inp.get("name") or inp.get("placeholder") for inp in inputs if inp]

    input_text = f"""
Instruction: {instruction}

Visible Buttons: {', '.join(buttons)}

Input Fields: {', '.join([str(f) for f in input_fields if f])}
""".strip()

    return input_text


def build_output_text(step):
    action = step["action"]

    action_type = action.get("type", "")
    target = action.get("target", "")
    value = action.get("value", "")

    if action_type == "TYPE_AND_CLICK_SEARCH":
        return f"TYPE search_box {value} -> CLICK search_button"

    elif action_type == "TYPE":
        return f"TYPE {target} {value}"

    elif action_type == "CLICK":
        return f"CLICK {target}"

    elif action_type == "STOP":
        return "STOP"

    return "UNKNOWN"


def transform_step(step):
    return {
        "input": build_input_text(step),
        "output": build_output_text(step)
    }