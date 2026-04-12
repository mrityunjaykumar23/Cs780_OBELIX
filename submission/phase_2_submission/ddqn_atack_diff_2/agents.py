import os
import numpy as np
import torch
import torch.nn as nn
from collections import deque

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

# =========================
# Q-Network (same as training)
# =========================
class QNet(nn.Module):
    def __init__(self, input_dim=72, hidden_dim=128, action_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# Globals
# =========================
_MODEL = None
_STACK = None
_STEP = 0


def _load_once():
    global _MODEL
    if _MODEL is not None:
        return

    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "weights1.pth")

    model = QNet()
    model.load_state_dict(torch.load(wpath, map_location="cpu"))
    model.eval()

    _MODEL = model


# =========================
# Policy
# =========================
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _MODEL, _STACK, _STEP
    _load_once()

    # ---------------------------
    # Reset stack at episode start
    # ---------------------------
    if _STEP == 0:
        _STACK = deque([obs]*4, maxlen=4)

    # ---------------------------
    # Update stack
    # ---------------------------
    _STACK.append(obs)
    state = np.concatenate(_STACK)

    x = torch.from_numpy(state.astype(np.float32))

    # ---------------------------
    # Forward pass
    # ---------------------------
    with torch.no_grad():
        q_vals = _MODEL(x)

    action = int(torch.argmax(q_vals).item())

    # ---------------------------
    # Step tracking
    # ---------------------------
    _STEP += 1

    if _STEP >= 2000:  # Codabench max_steps
        _STEP = 0

    return ACTIONS[action]