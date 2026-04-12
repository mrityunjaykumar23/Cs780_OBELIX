import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

from obelix import OBELIX

# ==============================
# Hyperparameters
# ==============================
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
BUFFER_SIZE = 50000
MIN_BUFFER = 5000

EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 0.995  # slow decay

TARGET_UPDATE = 500

EPISODES = 2000
MAX_STEPS = 1000

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
ACTION_SIZE = len(ACTIONS)


# ==============================
# Q Network
# ==============================
class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(18, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, ACTION_SIZE)
        )

    def forward(self, x):
        return self.net(x)


# ==============================
# Prioritized Replay Buffer
# ==============================
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(self, s, a, r, s2, d):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((s, a, r, s2, d))
        else:
            self.buffer[self.pos] = (s, a, r, s2, d)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        prios = self.priorities[:len(self.buffer)]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        samples = [self.buffer[i] for i in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        s, a, r, s2, d = zip(*samples)

        return (
            np.array(s),
            np.array(a),
            np.array(r, dtype=np.float32),
            np.array(s2),
            np.array(d, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32)
        )

    def update_priorities(self, indices, td_errors):
        for i, td in zip(indices, td_errors):
            self.priorities[i] = abs(td) + 1e-5

    def __len__(self):
        return len(self.buffer)


# ==============================
# Agent
# ==============================
class DDQNAgent:
    def __init__(self):
        self.q = QNet()
        self.target = QNet()
        self.target.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam(self.q.parameters(), lr=LR)
        self.buffer = PrioritizedReplayBuffer(BUFFER_SIZE)

        self.epsilon = EPS_START
        self.steps = 0

    def select_action(self, state):
        # exploration
        if random.random() < self.epsilon:
            probs = [0.1, 0.15, 0.5, 0.15, 0.1]  # bias forward
            return np.random.choice(ACTION_SIZE, p=probs)

        # exploitation
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_vals = self.q(s)
            return int(torch.argmax(q_vals).item())

    def train_step(self):
        if len(self.buffer) < MIN_BUFFER:
            return

        beta = min(1.0, 0.4 + self.steps * 1e-5)

        s, a, r, s2, d, indices, weights = self.buffer.sample(BATCH_SIZE, beta)

        s = torch.tensor(s, dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.long)
        r = torch.tensor(r)
        s2 = torch.tensor(s2, dtype=torch.float32)
        d = torch.tensor(d)
        weights = torch.tensor(weights)

        # current Q
        q_vals = self.q(s)
        q_val = q_vals.gather(1, a.unsqueeze(1)).squeeze(1)

        # DDQN target
        with torch.no_grad():
            next_actions = torch.argmax(self.q(s2), dim=1)
            next_q = self.target(s2).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = r + GAMMA * next_q * (1 - d)

        td_error = target - q_val

        loss = (weights * td_error.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update priorities
        self.buffer.update_priorities(indices, td_error.detach().numpy())

        # update target
        if self.steps % TARGET_UPDATE == 0:
            self.target.load_state_dict(self.q.state_dict())

        self.steps += 1


# ==============================
# Training Loop
# ==============================
def train():
    env = OBELIX(
        scaling_factor=5,
        max_steps=MAX_STEPS,
        wall_obstacles=True,
        difficulty=0  # LEVEL 0
    )

    agent = DDQNAgent()

    for ep in range(EPISODES):
        state = env.reset()
        total_reward = 0

        for t in range(MAX_STEPS):
            action_idx = agent.select_action(state)
            action = ACTIONS[action_idx]

            next_state, reward, done = env.step(action, render=False)

            # reward clipping
            reward = np.clip(reward, -10, 10)

            agent.buffer.push(state, action_idx, reward, next_state, done)
            agent.train_step()

            state = next_state
            total_reward += reward

            if done:
                break

        # epsilon decay PER EPISODE
        agent.epsilon = max(EPS_END, agent.epsilon * EPS_DECAY)

        print(f"Episode {ep}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    torch.save(agent.q.state_dict(), "weights.pth")
    print("✅ Model saved as weights.pth")


if __name__ == "__main__":
    train()