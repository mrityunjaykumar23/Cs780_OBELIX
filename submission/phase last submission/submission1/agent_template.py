from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
_model = None


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(18, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, len(ACTIONS)),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


def _load_once():
    global _model
    if _model is not None:
        return

    path = Path(__file__).with_name("weights.pth")
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    model = PolicyNet()
    model.load_state_dict(sd, strict=True)
    model.eval()
    _model = model


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    del rng
    _load_once()
    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    action = int(torch.argmax(_model(x), dim=-1).item())
    return ACTIONS[action]
