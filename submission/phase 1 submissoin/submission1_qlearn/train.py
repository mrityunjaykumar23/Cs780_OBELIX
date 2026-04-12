import numpy as np
import pickle
from obelix import OBELIX

# -----------------------
# Config
# -----------------------
ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

alpha = 0.1
gamma = 0.99

epsilon = 0.149
epsilon_decay = 0.999
epsilon_min = 0.05

num_episodes = 2000

# -----------------------
# Q-table
# -----------------------
try:
    with open("q_table.pkl", "rb") as f:
        Q = pickle.load(f)
    print("Loaded previous Q-table")
except:
    Q = {}
    print("Starting fresh")  # dict: state -> np.array(5)


def state_to_key(obs):
    return tuple(obs.astype(int))


def choose_action(state, epsilon):
    if state not in Q:
        Q[state] = np.zeros(5)

    if np.random.rand() < epsilon:
        return np.random.randint(5)
    else:
        return np.argmax(Q[state])


def update_q(state, action, reward, next_state):
    if next_state not in Q:
        Q[next_state] = np.zeros(5)

    best_next = np.max(Q[next_state])

    Q[state][action] += alpha * (
        reward + gamma * best_next - Q[state][action]
    )


# -----------------------
# Training Loop
# -----------------------
env = OBELIX(
    scaling_factor=5,
    arena_size=500,
    max_steps=800,
    wall_obstacles=True,
    difficulty=0,  # Level 1
)

for episode in range(num_episodes):
    obs = env.reset()
    state = state_to_key(obs)

    total_reward = 0
    done = False

    while not done:
        action_idx = choose_action(state, epsilon)
        action = ACTIONS[action_idx]

        next_obs, reward, done = env.step(action, render=False)
        next_state = state_to_key(next_obs)

        update_q(state, action_idx, reward, next_state)

        state = next_state
        total_reward += reward

    # decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # logging
    if episode % 100 == 0:
        print(f"Episode {episode}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

# -----------------------
# Save Q-table
# -----------------------
with open("q_table.pkl", "wb") as f:
    pickle.dump(Q, f)

print("Training complete. Q-table saved as q_table1.pkl")