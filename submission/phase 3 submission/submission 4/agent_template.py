from __future__ import annotations
from typing import List, Optional
import os
import collections
import numpy as np
import torch
import torch.nn as nn

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]
framestack = 32

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
        
class PolicyNet(nn.Module):
    def __init__(self, in_dim: int = 18 * framestack, n_actions: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(in_dim, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, n_actions), std=0.01),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

_model: Optional[PolicyNet] = None
_last_action: Optional[int] = None
_stack = collections.deque(maxlen=framestack)

def _load_once():
    global _model
    if _model is not None:
        return

    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")
    m = PolicyNet()
    sd = torch.load(wpath, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    m.load_state_dict(sd, strict=True)
    m.eval()
    _model = m

@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _last_action, _stack
    _load_once()
    if len(_stack) == 0:
        for _ in range(framestack):
            _stack.append(obs)
    else:
        _stack.append(obs)
    stacked_obs = np.concatenate(_stack)
    x = torch.tensor(stacked_obs, dtype=torch.float32).unsqueeze(0)
    logits = _model(x).squeeze(0).cpu().numpy()
    exp_logits = np.exp(logits - np.max(logits)) # stable softmax
    probs = exp_logits / np.sum(exp_logits)
    best = int(rng.choice(len(ACTIONS), p=probs))
    _last_action = best
    return ACTIONS[best]