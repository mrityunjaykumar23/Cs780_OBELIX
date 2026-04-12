"""Submission template.

Edit `policy()` to generate actions from an observation.
The evaluator will import this file and call `policy(obs, rng)`.

Action space (strings): 'L45', 'L22', 'FW', 'R22', 'R45'
Observation: numpy array shape (18,), values are 0/1.
"""

import numpy as np
import pickle
import os

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

Q = None
def state_to_key(obs):
    return tuple(obs.astype(int))

def load_q():
    global Q
    if Q is None:
        path = os.path.join(os.path.dirname(__file__), "q_table.pkl")
        with open(path, "rb") as f:
            Q = pickle.load(f)


def policy(obs, rng):
    load_q()

    state = state_to_key(obs)

    if state not in Q:
        return "FW"   # better than random

    return ACTIONS[int(np.argmax(Q[state]))]