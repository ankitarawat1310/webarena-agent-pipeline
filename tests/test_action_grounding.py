import numpy as np

from webarena_rl.training import action_to_idx, build_action_templates, decode_action, valid_action_mask


def _obs() -> dict:
    return {
        "instruction": "Find a wireless mouse under $50 and add it to cart.",
        "elements": [
            {"id": 1, "visible": True, "enabled": True, "is_text_input": True, "tag": "input", "role": "textbox", "text": "", "href": "", "class_name": ""},
            {"id": 2, "visible": True, "enabled": True, "is_text_input": False, "tag": "a", "role": "link", "text": "Wireless Mouse Pro", "href": "/mouse", "class_name": ""},
            {"id": 3, "visible": True, "enabled": False, "is_text_input": True, "tag": "input", "role": "textbox", "text": "", "href": "", "class_name": ""},
        ],
    }


def test_action_template_size_and_decode_slot_mapping() -> None:
    templates = build_action_templates(3)
    assert len(templates) == 2 * 3 + 4
    click_idx = templates.index(("click", 0))
    type_idx = templates.index(("type", 0))
    click_action = decode_action(click_idx, _obs(), templates)
    type_action = decode_action(type_idx, _obs(), templates)
    assert click_action["element_id"] == 2
    assert type_action["element_id"] == 1
    assert "wireless mouse" in type_action["text"].lower()


def test_valid_action_mask_basics() -> None:
    templates = build_action_templates(2)
    mask = valid_action_mask(_obs(), templates)
    assert mask[templates.index(("type", 0))]
    assert not mask[templates.index(("type", 1))]
    assert mask[templates.index(("scroll", None))]
    assert mask[templates.index(("go_back", None))]
    assert mask[templates.index(("wait", None))]
    assert mask[templates.index(("stop", None))]


def test_action_to_idx_mapping_and_wait_fallback() -> None:
    templates = build_action_templates(3)
    state = _obs()
    idx_click = action_to_idx({"action_type": "click", "element_id": 2}, state, templates)
    assert templates[idx_click] == ("click", 0)
    idx_missing = action_to_idx({"action_type": "click", "element_id": 999}, state, templates)
    assert templates[idx_missing] == ("wait", None)
