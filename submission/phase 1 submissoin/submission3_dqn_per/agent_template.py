"""Submission template.

Edit `policy()` to generate actions from an observation.
The evaluator will import this file and call `policy(obs, rng)`.

Action space (strings): 'L45', 'L22', 'FW', 'R22', 'R45'
Observation: numpy array shape (18,), values are 0/1.
"""

import numpy as np
import pickle
import os
import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

_MODEL = None

def _load():
    global _MODEL
    if _MODEL is not None:
        return


    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(18, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 5),
            )

        def forward(self, x):
            return self.net(x)

    model = Net()
    path = os.path.join(os.path.dirname(__file__), "weights.pth")
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()

    _MODEL = model


def policy(obs, rng):
    _load()

    import torch
    x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        q = _MODEL(x).squeeze(0).numpy()

    return ACTIONS[int(np.argmax(q))]