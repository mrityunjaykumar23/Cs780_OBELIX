from __future__ import annotations

import argparse
import collections
import importlib.util
import multiprocessing as mp
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
ACTION_COUNT = len(ACTIONS)
OBSERVATION_SIZE = 18


def parse_int_list(text: str) -> list[int]:
    return [int(chunk.strip()) for chunk in text.split(",") if chunk.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(chunk.strip()) for chunk in text.split(",") if chunk.strip()]


@dataclass(frozen=True)
class CurriculumStage:
    difficulty: int
    wall_obstacles: bool
    start: int
    end: int


class CurriculumSchedule:
    def __init__(self, stages: list[CurriculumStage]):
        if not stages:
            raise ValueError("curriculum must contain at least one stage")
        self.stages = stages

    @classmethod
    def from_episode_budget(
        cls,
        total_episodes: int,
        difficulties: list[int],
        ratios: list[float],
    ) -> "CurriculumSchedule":
        if len(difficulties) != len(ratios):
            raise ValueError("curriculum difficulties and ratios must have the same length")

        total_ratio = float(sum(ratios))
        if total_ratio <= 0.0:
            raise ValueError("curriculum ratios must sum to a positive value")

        normalized = [ratio / total_ratio for ratio in ratios]
        stage_lengths = [int(total_episodes * fraction) for fraction in normalized]
        stage_lengths[-1] += total_episodes - sum(stage_lengths)

        stages: list[CurriculumStage] = []
        cursor = 0
        for difficulty, stage_len in zip(difficulties, stage_lengths):
            stages.append(
                CurriculumStage(
                    difficulty=difficulty,
                    wall_obstacles=True,
                    start=cursor,
                    end=cursor + stage_len,
                )
            )
            cursor += stage_len

        return cls(stages)

    def stage_for_episode(self, episode_idx: int) -> CurriculumStage:
        for stage in self.stages:
            if episode_idx < stage.end:
                return stage
        return self.stages[-1]


def load_environment_class(obelix_py: str):
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load environment from {obelix_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OBELIX


def env_worker(connection, obelix_py: str, base_env_kwargs: dict, stuck_limit: int):
    obelix_cls = load_environment_class(obelix_py)
    env = None
    stuck_steps = 0

    while True:
        command, payload = connection.recv()

        if command == "reset":
            if isinstance(payload, dict):
                episode_seed = int(payload["seed"])
                runtime_kwargs = dict(base_env_kwargs)
                runtime_kwargs.update(payload.get("env_kwargs", {}))
            else:
                episode_seed = int(payload)
                runtime_kwargs = dict(base_env_kwargs)

            stuck_steps = 0
            env = obelix_cls(**runtime_kwargs, seed=episode_seed)
            initial_obs = np.asarray(env.reset(seed=episode_seed), dtype=np.float32)
            connection.send(initial_obs)
            continue

        if command == "step":
            next_obs, reward, done = env.step(ACTIONS[payload], render=False)
            next_obs = np.asarray(next_obs, dtype=np.float32)

            if next_obs[17] > 0.5:
                stuck_steps += 1
            else:
                stuck_steps = 0

            if stuck_steps >= stuck_limit:
                done = True
                reward -= 100.0
                stuck_steps = 0

            connection.send((next_obs, float(reward), bool(done)))
            continue

        if command == "close":
            connection.close()
            return

        raise ValueError(f"Unknown worker command: {command}")


class ParallelObelixEnv:
    def __init__(
        self,
        obelix_py: str,
        num_envs: int,
        env_kwargs: dict,
        *,
        base_seed: int,
        stuck_limit: int = 20,
    ):
        self.num_envs = num_envs
        self.base_seed = base_seed
        self.parent_conns, child_conns = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.processes: list[mp.Process] = []

        for child_conn in child_conns:
            proc = mp.Process(
                target=env_worker,
                args=(child_conn, obelix_py, env_kwargs, stuck_limit),
                daemon=True,
            )
            proc.start()
            self.processes.append(proc)

    def reset_all(self, episode_index: int, env_override: dict | None = None) -> np.ndarray:
        override = env_override or {}
        for env_idx, conn in enumerate(self.parent_conns):
            conn.send(
                (
                    "reset",
                    {
                        "seed": self.base_seed + episode_index * self.num_envs + env_idx,
                        "env_kwargs": override,
                    },
                )
            )
        return np.stack([conn.recv() for conn in self.parent_conns])

    def step_all(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        for conn, action in zip(self.parent_conns, actions):
            conn.send(("step", int(action)))

        transitions = [conn.recv() for conn in self.parent_conns]
        next_obs, rewards, dones = zip(*transitions)
        return (
            np.stack(next_obs),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
        )

    def reset_single(self, env_index: int, seed: int, env_override: dict | None = None) -> np.ndarray:
        self.parent_conns[env_index].send(
            (
                "reset",
                {
                    "seed": int(seed),
                    "env_kwargs": env_override or {},
                },
            )
        )
        return self.parent_conns[env_index].recv()

    def close(self) -> None:
        for conn in self.parent_conns:
            conn.send(("close", None))
        for proc in self.processes:
            proc.join(timeout=5)


class RandomNetworkDistillation(nn.Module):
    def __init__(
        self,
        input_dim: int = OBSERVATION_SIZE,
        output_dim: int = 64,
        lr: float = 1e-4,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, output_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, output_dim),
        )
        for parameter in self.target.parameters():
            parameter.requires_grad = False

        self.to(device)
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=lr)
        self.running_mean = 0.0
        self.running_var = 1.0
        self.running_count = 1

    def surprise(self, obs_batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            target_features = self.target(obs_batch)
        predicted_features = self.predictor(obs_batch)
        return F.mse_loss(predicted_features, target_features.detach(), reduction="none").mean(dim=1)

    def train_predictor(self, obs_batch: torch.Tensor) -> None:
        with torch.no_grad():
            target_features = self.target(obs_batch)
        predicted_features = self.predictor(obs_batch)
        loss = F.mse_loss(predicted_features, target_features)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def normalized_reward(self, obs_batch: torch.Tensor, beta: float) -> torch.Tensor:
        raw_reward = self.surprise(obs_batch)
        self.running_count += raw_reward.numel()
        self.running_mean += (raw_reward.mean().item() - self.running_mean) / self.running_count
        self.running_var = max(
            self.running_var + ((raw_reward - self.running_mean) ** 2).mean().item() / self.running_count,
            1e-6,
        )
        return beta * (raw_reward - self.running_mean) / np.sqrt(self.running_var)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, ACTION_COUNT),
        )

    def forward(self, obs_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(obs_batch)
        return F.softmax(logits, dim=-1), F.log_softmax(logits, dim=-1)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, ACTION_COUNT),
        )

    def forward(self, obs_batch: torch.Tensor) -> torch.Tensor:
        return self.net(obs_batch)


class ReplayMemory:
    def __init__(self, capacity: int = 300_000):
        self._storage = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done) -> None:
        self._storage.append((state, int(action), float(reward), next_state, float(done)))

    def sample(self, batch_size: int):
        batch = random.sample(self._storage, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.asarray(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.asarray(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self._storage)


class DiscreteSACAgent:
    def __init__(self, *, lr: float, gamma: float, tau: float = 0.005, device: torch.device):
        self.gamma = gamma
        self.tau = tau
        self.device = device

        self.actor = Actor().to(device)
        self.critic_a = Critic().to(device)
        self.critic_b = Critic().to(device)
        self.target_critic_a = Critic().to(device)
        self.target_critic_b = Critic().to(device)
        self.target_critic_a.load_state_dict(self.critic_a.state_dict())
        self.target_critic_b.load_state_dict(self.critic_b.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(
            list(self.critic_a.parameters()) + list(self.critic_b.parameters()),
            lr=lr,
        )

        self.target_entropy = -np.log(1.0 / ACTION_COUNT) * 0.5
        self.log_alpha = torch.zeros(1, device=device, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.alpha = self.log_alpha.exp().item()

    @torch.no_grad()
    def choose_actions(self, obs_batch: np.ndarray) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        action_probs, _ = self.actor(obs_tensor)
        distribution = torch.distributions.Categorical(action_probs)
        return distribution.sample().cpu().numpy()

    def learn(self, replay: ReplayMemory, batch_size: int, rnd: RandomNetworkDistillation, rnd_beta: float) -> None:
        states, actions, rewards, next_states, dones = replay.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        intrinsic_bonus = rnd.normalized_reward(next_states, rnd_beta)
        rnd.train_predictor(next_states)
        total_reward = rewards + intrinsic_bonus.detach()

        with torch.no_grad():
            next_probs, next_log_probs = self.actor(next_states)
            target_q_values = torch.min(
                self.target_critic_a(next_states),
                self.target_critic_b(next_states),
            )
            next_value = (next_probs * (target_q_values - self.alpha * next_log_probs)).sum(dim=1)
            td_target = total_reward + self.gamma * (1.0 - dones) * next_value

        predicted_q_a = self.critic_a(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        predicted_q_b = self.critic_b(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        critic_loss = F.mse_loss(predicted_q_a, td_target) + F.mse_loss(predicted_q_b, td_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        action_probs, log_action_probs = self.actor(states)
        critic_min = torch.min(self.critic_a(states), self.critic_b(states))
        actor_loss = (action_probs * (self.alpha * log_action_probs - critic_min)).sum(dim=1).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        entropy = -(action_probs * log_action_probs).sum(dim=1).detach()
        alpha_loss = (self.log_alpha * (entropy - self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.log_alpha.data.clamp_(-3.0, 0.5)
        self.alpha = self.log_alpha.exp().item()

        self._soft_update(self.target_critic_a, self.critic_a)
        self._soft_update(self.target_critic_b, self.critic_b)

    def _soft_update(self, target_network: nn.Module, source_network: nn.Module) -> None:
        for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(self.tau * source_param.data)


@dataclass
class TrainingConfig:
    obelix_py: str
    out: str
    episodes: int
    max_steps: int
    num_envs: int
    difficulty: int
    wall_obstacles: bool
    box_speed: int
    scaling_factor: int
    arena_size: int
    gamma: float
    lr: float
    batch: int
    replay_start: int
    updates_per_step: int
    rnd_beta: float
    load_weights: str
    seed: int
    curriculum: bool
    curriculum_difficulties: str
    curriculum_ratios: str


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vectorized SAC+RND for OBELIX")
    parser.add_argument("--obelix_py", type=str, required=True)
    parser.add_argument("--out", type=str, default="sac_vec.pth")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument(
        "--num_envs",
        type=int,
        default=4,
        help="Number of parallel environment workers. Match to CPU core count.",
    )
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--box_speed", type=int, default=2)
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--batch",
        type=int,
        default=128,
        help="Larger batch keeps GPU busy. Scales with num_envs.",
    )
    parser.add_argument("--replay_start", type=int, default=12000)
    parser.add_argument("--updates_per_step", type=int, default=1)
    parser.add_argument("--rnd_beta", type=float, default=0.02)
    parser.add_argument(
        "--load_weights",
        type=str,
        default="",
        help="Path to pre-trained actor weights for curriculum learning.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Use a built-in curriculum that progresses through easier difficulties.",
    )
    parser.add_argument(
        "--curriculum_difficulties",
        type=str,
        default="0,2,3",
        help="Comma-separated difficulty stages used when --curriculum is set.",
    )
    parser.add_argument(
        "--curriculum_ratios",
        type=str,
        default="0.25,0.35,0.40",
        help="Comma-separated stage fractions used when --curriculum is set.",
    )
    parser.set_defaults(curriculum=True, wall_obstacles=True)
    return parser


def make_training_config(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        obelix_py=args.obelix_py,
        out=args.out,
        episodes=args.episodes,
        max_steps=args.max_steps,
        num_envs=args.num_envs,
        difficulty=args.difficulty,
        wall_obstacles=args.wall_obstacles,
        box_speed=args.box_speed,
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        gamma=args.gamma,
        lr=args.lr,
        batch=args.batch,
        replay_start=args.replay_start,
        updates_per_step=args.updates_per_step,
        rnd_beta=args.rnd_beta,
        load_weights=args.load_weights,
        seed=args.seed,
        curriculum=args.curriculum,
        curriculum_difficulties=args.curriculum_difficulties,
        curriculum_ratios=args.curriculum_ratios,
    )


def make_env_kwargs(config: TrainingConfig) -> dict:
    return {
        "scaling_factor": config.scaling_factor,
        "arena_size": config.arena_size,
        "max_steps": config.max_steps,
        "wall_obstacles": config.wall_obstacles,
        "difficulty": config.difficulty,
        "box_speed": config.box_speed,
    }


def load_actor_weights_if_requested(agent: DiscreteSACAgent, weights_path: str, device: torch.device) -> None:
    if not weights_path:
        return
    state_dict = torch.load(weights_path, map_location=device)
    agent.actor.load_state_dict(state_dict, strict=True)
    print(f"Loaded actor weights from {weights_path} (curriculum mode)")


def print_curriculum(schedule: CurriculumSchedule) -> None:
    print("Curriculum plan:")
    for stage_number, stage in enumerate(schedule.stages, start=1):
        print(
            f"  Stage {stage_number}: episodes {stage.start + 1}-{stage.end} "
            f"difficulty={stage.difficulty} wall_obstacles={stage.wall_obstacles}"
        )


def stage_override(stage: CurriculumStage | None) -> dict | None:
    if stage is None:
        return None
    return {
        "difficulty": stage.difficulty,
        "wall_obstacles": stage.wall_obstacles,
    }


def main() -> None:
    mp.set_start_method("spawn", force=True)

    args = build_arg_parser().parse_args()
    config = make_training_config(args)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Parallel envs: {config.num_envs}")

    curriculum = None
    if config.curriculum:
        curriculum = CurriculumSchedule.from_episode_budget(
            total_episodes=config.episodes,
            difficulties=parse_int_list(config.curriculum_difficulties),
            ratios=parse_float_list(config.curriculum_ratios),
        )
        print_curriculum(curriculum)

    vec_env = ParallelObelixEnv(
        config.obelix_py,
        config.num_envs,
        make_env_kwargs(config),
        base_seed=config.seed,
    )
    agent = DiscreteSACAgent(lr=config.lr, gamma=config.gamma, device=device)
    load_actor_weights_if_requested(agent, config.load_weights, device)
    rnd = RandomNetworkDistillation(device=device)
    replay = ReplayMemory()

    episode_returns = np.zeros(config.num_envs, dtype=np.float32)
    completed_episodes = 0
    successful_episodes = 0
    best_episode_return = -float("inf")
    last_log_checkpoint = -1
    recent_returns: list[float] = []
    recent_successes: list[int] = []

    first_stage = None if curriculum is None else curriculum.stage_for_episode(0)
    observations = vec_env.reset_all(episode_index=0, env_override=stage_override(first_stage))

    try:
        while completed_episodes < config.episodes:
            if len(replay) < config.replay_start:
                chosen_actions = np.random.randint(0, ACTION_COUNT, size=config.num_envs)
            else:
                chosen_actions = agent.choose_actions(observations)

            next_observations, rewards, dones = vec_env.step_all(chosen_actions)

            for env_idx in range(config.num_envs):
                replay.add(
                    observations[env_idx],
                    chosen_actions[env_idx],
                    rewards[env_idx],
                    next_observations[env_idx],
                    dones[env_idx],
                )
                episode_returns[env_idx] += rewards[env_idx]

                if not dones[env_idx]:
                    continue

                success = int(rewards[env_idx] > 1000.0)
                completed_return = float(episode_returns[env_idx])
                recent_returns.append(completed_return)
                recent_successes.append(success)
                if len(recent_returns) > 50:
                    recent_returns.pop(0)
                    recent_successes.pop(0)

                if completed_return > best_episode_return:
                    best_episode_return = completed_return
                    torch.save(agent.actor.state_dict(), config.out)

                successful_episodes += success
                episode_returns[env_idx] = 0.0
                completed_episodes += 1

                next_stage = None if curriculum is None else curriculum.stage_for_episode(completed_episodes)
                next_observations[env_idx] = vec_env.reset_single(
                    env_index=env_idx,
                    seed=config.seed + completed_episodes + env_idx,
                    env_override=stage_override(next_stage) or {},
                )

                if completed_episodes >= config.episodes:
                    break

            observations = next_observations

            if len(replay) >= config.replay_start:
                for _ in range(config.updates_per_step):
                    agent.learn(replay, config.batch, rnd, config.rnd_beta)

            should_log = (
                completed_episodes > 0
                and completed_episodes % 50 == 0
                and completed_episodes != last_log_checkpoint
                and recent_returns
            )
            if should_log:
                last_log_checkpoint = completed_episodes
                active_stage = None if curriculum is None else curriculum.stage_for_episode(completed_episodes)
                extra_stage_info = ""
                if active_stage is not None:
                    extra_stage_info = (
                        f" stage_difficulty={active_stage.difficulty}"
                        f" stage_walls={int(active_stage.wall_obstacles)}"
                    )

                print(
                    f"Ep {completed_episodes}/{config.episodes} "
                    f"mean={np.mean(recent_returns):.1f} "
                    f"max={np.max(recent_returns):.1f} "
                    f"succ_50={int(np.sum(recent_successes))} "
                    f"alpha={agent.alpha:.3f} "
                    f"buf={len(replay)} "
                    f"best={best_episode_return:.1f}"
                    f"{extra_stage_info}"
                )
    finally:
        vec_env.close()

    print(f"\nDone! Successes: {successful_episodes}/{config.episodes}")
    torch.save(agent.actor.state_dict(), config.out)
    print(f"Saved actor -> {config.out}")


if __name__ == "__main__":
    main()
