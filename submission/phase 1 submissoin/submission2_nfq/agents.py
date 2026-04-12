"""Submission template for NFQ Agent.

Edit `policy()` to generate actions from an observation.
The evaluator will import this file and call `policy(obs, rng)`.

Action space (strings): 'L45', 'L22', 'FW', 'R22', 'R45'
Observation: numpy array shape (18,), values are 0/1.
"""

import os
import numpy as np

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

_MODEL = None  

def _load_once():
    """Load the trained model and weights lazily."""
    global _MODEL
    if _MODEL is not None:
        return

    import torch
    import torch.nn as nn

    class QNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(18, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 5)
            )

        def forward(self, x):
            return self.net(x)

    submission_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(submission_dir, "weights.pth")
    model = QNet()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    _MODEL = model

def policy(obs, rng):
    """Use the trained model to choose the best action."""
    _load_once()
    
    import torch
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    with torch.no_grad():
        q = _MODEL(obs_tensor)
    action = torch.argmax(q).item()
    return ACTIONS[action]