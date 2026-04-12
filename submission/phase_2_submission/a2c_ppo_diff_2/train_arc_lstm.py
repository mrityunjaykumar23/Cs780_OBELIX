import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from obelix import OBELIX

# Actions
ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

device = torch.device("cpu")

# =========================
# Model: A2C + LSTM
# =========================
class A2CLSTM(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=128, action_dim=5):
        super().__init__()

        self.fc = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x, hx, cx):
        # x: (1, 1, 18)
        x = torch.relu(self.fc(x))
        out, (hx, cx) = self.lstm(x, (hx, cx))

        out = out[:, -1, :]  # last timestep

        logits = self.actor(out)
        value = self.critic(out)

        return logits, value, hx, cx


# =========================
# Hyperparameters
# =========================
gamma = 0.995
lr = 1e-3
num_episodes = 500
max_steps = 1000

model = A2CLSTM().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)


# =========================
# Training loop
# =========================
for episode in range(num_episodes):

    env = OBELIX(
        scaling_factor=5,
        arena_size=500,
        max_steps=max_steps,
        wall_obstacles=True,
        difficulty=2,   # blinking box
    )

    obs = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    hx = torch.zeros(1, 1, 128)
    cx = torch.zeros(1, 1, 128)

    log_probs = []
    values = []
    rewards = []

    total_reward = 0

    for step in range(max_steps):

        logits, value, hx, cx = model(obs, hx, cx)

        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        action_str = ACTIONS[action.item()]

        next_obs, reward, done = env.step(action_str, render=False)

        total_reward += reward

        log_probs.append(log_prob)
        values.append(value.squeeze())
        rewards.append(reward)

        obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        if done:
            break

    # =========================
    # Compute returns
    # =========================
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32)

    values = torch.stack(values)
    log_probs = torch.stack(log_probs)

    advantage = returns - values.detach()

    # =========================
    # Loss
    # =========================
    actor_loss = -(log_probs * advantage).mean()
    critic_loss = (returns - values).pow(2).mean()

    loss = actor_loss + 0.5 * critic_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Episode {episode}, Reward: {total_reward:.2f}")

# =========================
# Save weights
# =========================
torch.save(model.state_dict(), "weights.pth")