
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from obelix import OBELIX

# ========================
# Hyperparameters
# ========================
LR = 3e-4
GAMMA = 0.99
CLIP_EPS = 0.2
EPOCHS = 6
ENTROPY_COEF = 0.03
UPDATE_TIMESTEP = 4000
EPISODES = 1000

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

# ========================
# Actor-Critic Network
# ========================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim=18, action_dim=5):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

# ========================
# Rollout Buffer
# ========================
class Buffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def clear(self):
        self.__init__()

# ========================
# Helper functions
# ========================
def select_action(model, obs):
    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    logits, value = model(obs)

    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)

    action = dist.sample()
    return action.item(), dist.log_prob(action), value.item()

def compute_returns_advantages(rewards, values, dones):
    returns = []
    G = 0

    for r, d in zip(reversed(rewards), reversed(dones)):
        if d:
            G = 0
        G = r + GAMMA * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32)
    values = torch.tensor(values, dtype=torch.float32)

    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return returns, advantages

def ppo_update(model, optimizer, buffer):
    states = torch.tensor(buffer.states, dtype=torch.float32)
    actions = torch.tensor(buffer.actions)
    old_logprobs = torch.stack(buffer.logprobs).detach()

    returns, advantages = compute_returns_advantages(
        buffer.rewards, buffer.values, buffer.dones
    )

    for _ in range(EPOCHS):
        logits, values = model(states)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        logprobs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratios = torch.exp(logprobs - old_logprobs)

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values.squeeze(), returns)

        loss = policy_loss + 0.5 * value_loss - ENTROPY_COEF * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# ========================
# Training
# ========================
env = OBELIX(difficulty=3,scaling_factor=5, wall_obstacles=True)

model = ActorCritic()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
buffer = Buffer()

timestep = 0
last_actions = []

for ep in range(EPISODES):
    obs = env.reset()
    done = False
    ep_reward = 0

    while not done:
        action, logprob, value = select_action(model, obs)

        # ========================
        # Rotation limiter (CRITICAL)
        # ========================
        last_actions.append(action)
        if len(last_actions) > 5:
            last_actions.pop(0)

        if len(last_actions) == 5 and all(a != 2 for a in last_actions):
            action = 2  # force FW

        action_str = ACTIONS[action]

        next_obs, reward, done = env.step(action_str, render=False)

        buffer.states.append(obs)
        buffer.actions.append(action)
        buffer.logprobs.append(logprob)
        buffer.rewards.append(reward)
        buffer.values.append(value)
        buffer.dones.append(done)

        obs = next_obs
        ep_reward += reward
        timestep += 1

        if timestep % UPDATE_TIMESTEP == 0:
            ppo_update(model, optimizer, buffer)
            buffer.clear()

    print(f"Episode {ep}, Reward: {ep_reward}")

# ========================
# Save model
# ========================
torch.save(model.state_dict(), "ppo_obelix.pth")
