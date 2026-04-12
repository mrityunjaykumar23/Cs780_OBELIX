import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS = ("L45", "L22", "FW", "R22", "R45")
class A2CLSTM(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=128, action_dim=5):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1) 

    def forward(self, x, hx, cx):
        x = torch.relu(self.fc(x))
        out, (hx, cx) = self.lstm(x, (hx, cx))
        out = out[:, -1, :]
        logits = self.actor(out)
        value = self.critic(out)  
        return logits, value, hx, cx

_MODEL = None
_HX = None
_CX = None
_STEP = 0
def _load_once():
    global _MODEL
    if _MODEL is not None:
        return
    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "weights.pth")
    model = A2CLSTM()
    model.load_state_dict(torch.load(wpath, map_location="cpu"))
    model.eval()
    _MODEL = model

def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _MODEL, _HX, _CX, _STEP
    _load_once()

    # Reset LSTM at new episode
    if _STEP == 0:
        _HX = torch.zeros(1, 1, 128)
        _CX = torch.zeros(1, 1, 128)

    x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        logits, _, _HX, _CX = _MODEL(x, _HX, _CX)

    probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()

    if rng.random() < 0.1:
        action = rng.choice(len(ACTIONS))   # small exploration
    else:
        action = int(np.argmax(probs))

    if np.sum(obs) == 0:
        if rng.random() < 0.6:
            action = 2  # FW (move forward)
        else:
            action = rng.choice([1, 3])  # L22 or R22
    if np.sum(obs) > 0:
        # front sensors (approx indices)
        if obs[8] == 1 or obs[9] == 1:
            action = 2  # go forward

    _STEP += 1
    if _STEP > 2000:
        _STEP = 0

    return ACTIONS[action]