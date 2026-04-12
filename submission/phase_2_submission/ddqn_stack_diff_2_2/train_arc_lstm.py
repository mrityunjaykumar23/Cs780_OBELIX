import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

from obelix import OBELIX

device = torch.device("cpu")

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
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

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, d = zip(*batch)
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(s_next, dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)
    
def init_stack(obs, k=4):
    return deque([obs]*k, maxlen=k)

def get_state(stack):
    return np.concatenate(stack)

gamma = 0.99
lr = 1e-4
batch_size = 64
epsilon = 0.3
epsilon_min = 0.05
epsilon_decay = 0.999
target_update = 200

num_episodes = 300
max_steps = 1000
q_net = QNet().to(device)
target_net = QNet().to(device)
q_net.load_state_dict(torch.load("weights1_1.pth"))
# target_net.load_state_dict(q_net.state_dict())
target_net.load_state_dict(q_net.state_dict())

optimizer = optim.Adam(q_net.parameters(), lr=lr)
buffer = ReplayBuffer()

step_count = 0

for episode in range(num_episodes):

    env = OBELIX(
        scaling_factor=5,
        arena_size=500,
        max_steps=max_steps,
        wall_obstacles=True,
        difficulty=2
    )

    obs = env.reset()
    stack = init_stack(obs)
    state = get_state(stack)

    total_reward = 0

    for t in range(max_steps):

        # ε-greedy
        if random.random() < epsilon:
            action = random.randint(0, 4)
        else:
            with torch.no_grad():
                q_vals = q_net(torch.tensor(state, dtype=torch.float32))
                action = torch.argmax(q_vals).item()

        next_obs, reward, done = env.step(ACTIONS[action], render=False)

        stack.append(next_obs)
        next_state = get_state(stack)

        buffer.push(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward
        step_count += 1

        # training
        if len(buffer) > batch_size:
            s, a, r, s_next, d = buffer.sample(batch_size)

            q_values = q_net(s)
            q_val = q_values.gather(1, a.unsqueeze(1)).squeeze()

            # DDQN target
            next_actions = torch.argmax(q_net(s_next), dim=1)
            next_q = target_net(s_next).gather(1, next_actions.unsqueeze(1)).squeeze()

            target = r + gamma * next_q * (1 - d)

            loss = nn.MSELoss()(q_val, target.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # update target
        if step_count % target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

        if done:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    print(f"Episode {episode}, Reward: {total_reward:.2f}, Eps: {epsilon:.3f}")
torch.save(q_net.state_dict(), "weights1_2.pth")