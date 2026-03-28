from copy import deepcopy


REQUIRED_FIELDS = [
    "task_id",
    "episode_id",
    "step_id",
    "instruction",
    "observation_before",
    "action",
    "observation_after",
    "reward",
    "done",
    "timestamp",
]


def standardize_action(action):
    action = deepcopy(action)

    if "type" in action and isinstance(action["type"], str):
        action["type"] = action["type"].strip().upper()

    if "target" not in action:
        action["target"] = ""

    if "value" not in action:
        action["value"] = ""

    return action


def is_valid_step(step):
    for field in REQUIRED_FIELDS:
        if field not in step:
            return False
    return True


def preprocess_step(step):
    cleaned = deepcopy(step)

    cleaned["instruction"] = str(cleaned["instruction"]).strip()
    cleaned["action"] = standardize_action(cleaned["action"])

    if cleaned["reward"] is None:
        cleaned["reward"] = 0

    cleaned["done"] = bool(cleaned["done"])

    return cleaned