import numpy as np
import torch
import torch.nn as nn

from webarena_rl.training import _compute_td_loss, build_action_templates


class FixedQ(nn.Module):
    def __init__(self, table: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("table", table)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = x[:, 0].long()
        return self.table[idx]


def _state(kind: int) -> dict:
    if kind == 0:
        elems = [{"id": 1, "visible": True, "enabled": True, "is_text_input": False, "tag": "a", "role": "link", "text": "item", "href": "/p", "class_name": ""}]
    else:
        elems = [{"id": 2, "visible": True, "enabled": True, "is_text_input": True, "tag": "input", "role": "textbox", "text": "", "href": "", "class_name": ""}]
    return {"elements": elems, "instruction": "Find laptop sleeve under $40 and add to cart.", "kind": kind}


def _encoder(obs: dict) -> np.ndarray:
    return np.array([float(obs["kind"]), 0.0], dtype=np.float32)


def test_td_loss_runs_for_dqn_and_ddqn_with_masks() -> None:
    templates = build_action_templates(2)
    out_dim = len(templates)
    table = torch.zeros((2, out_dim), dtype=torch.float32)
    table[:, :] = -1.0
    table[1, templates.index(("click", 1))] = 100.0  # invalid in next state
    table[1, templates.index(("wait", None))] = 1.0
    online = FixedQ(table.clone())
    target = FixedQ(table.clone())
    batch = [
        {
            "state": _state(0),
            "action": {"action_type": "click", "element_id": 1},
            "reward": 0.0,
            "next_state": _state(1),
            "done": False,
        }
    ]
    loss_dqn, invalid_dqn = _compute_td_loss(batch, online, target, gamma=0.99, device=torch.device("cpu"), ddqn=False, encoder_fn=_encoder, action_templates=templates)
    loss_ddqn, invalid_ddqn = _compute_td_loss(batch, online, target, gamma=0.99, device=torch.device("cpu"), ddqn=True, encoder_fn=_encoder, action_templates=templates)
    assert torch.isfinite(loss_dqn)
    assert torch.isfinite(loss_ddqn)
    assert invalid_dqn == 0
    assert invalid_ddqn == 0
