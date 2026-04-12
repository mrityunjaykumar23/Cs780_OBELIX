import os
import numpy as np

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

_MODEL = None
_last_actions = []

def _load_once():
    global _MODEL
    if _MODEL is not None:
        return

    import torch
    import torch.nn as nn

    class ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(18, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
            )
            self.actor = nn.Linear(128, 5)
            self.critic = nn.Linear(128, 1)

        def forward(self, x):
            x = self.shared(x)
            logits = self.actor(x)
            value = self.critic(x)
            return logits, value

    model = ActorCritic()

    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "ppo_obelix.pth")

    import torch
    state_dict = torch.load(wpath, map_location="cpu")
    model.load_state_dict(state_dict)

    model.eval()
    _MODEL = model


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _last_actions

    _load_once()

    import torch

    x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        logits, _ = _MODEL(x)
        logits = logits.squeeze(0).numpy()
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    sum_exp = np.sum(exp_logits)

    if sum_exp == 0 or np.isnan(sum_exp) or np.any(np.isnan(exp_logits)):
        probs = np.ones(len(ACTIONS)) / len(ACTIONS)
    else:
        probs = exp_logits / sum_exp

    if np.any(np.isnan(probs)) or np.sum(probs) == 0:
        probs = np.ones(len(ACTIONS)) / len(ACTIONS)
    action = int(rng.choice(len(ACTIONS), p=probs))
    _last_actions.append(action)
    if len(_last_actions) > 6:
        _last_actions.pop(0)

    if len(_last_actions) == 6 and all(a != 2 for a in _last_actions):
        action = 2  # force "FW"

    return ACTIONS[action]
