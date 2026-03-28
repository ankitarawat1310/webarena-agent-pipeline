import random

SEARCH_TERMS = ["phone", "laptop", "shirt", "watch", "bag"]

def choose_action(observation, instruction):
    instruction_lower = instruction.lower()

    if "search" in instruction_lower and observation.get("search_box_present", False):
        return {
            "type": "TYPE_AND_CLICK_SEARCH",
            "target": 'input[name="q"] + Search button',
            "value": random.choice(SEARCH_TERMS)
        }

    return {
        "type": "STOP",
        "target": "",
        "value": ""
    }