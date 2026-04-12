# Run example:
#   python train.py --obelix_py obelix.py
#
# How parallelism works:
#   N worker processes each run their own OBELIX copy.
#   Every step: main process collects N observations simultaneously,
#   batches them into one GPU call, and adds N transitions to the replay buffer.
#   The GPU now gets N-times larger batches per wall-clock second.

from __future__ import annotations
import argparse, random, os, collections
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = len(ACTIONS)
OBS_DIM   = 18


def _parse_int_csv(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_float_csv(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _build_curriculum_plan(total_episodes: int, difficulties: list[int], ratios: list[float]) -> list[dict]:
    if len(difficulties) != len(ratios):
        raise ValueError("curriculum difficulties and ratios must have the same length")
    if len(difficulties) == 0:
        raise ValueError("curriculum must contain at least one stage")

    ratio_sum = float(sum(ratios))
    if ratio_sum <= 0.0:
        raise ValueError("curriculum ratios must sum to a positive value")
    normalized = [r / ratio_sum for r in ratios]

    stage_counts = [int(total_episodes * frac) for frac in normalized]
    stage_counts[-1] += total_episodes - sum(stage_counts)

    plan = []
    start = 0
    for difficulty, count in zip(difficulties, stage_counts):
        end = start + count
        plan.append(
            {
                "difficulty": difficulty,
                "wall_obstacles": True,
                "start": start,
                "end": end,
            }
        )
        start = end
    return plan


def _curriculum_stage_for_episode(plan: list[dict], episode_idx: int) -> dict:
    for stage in plan:
        if episode_idx < stage["end"]:
            return stage
    return plan[-1]

# =====================================================================
# SUBPROCESS ENVIRONMENT WORKER
# Each worker lives in its own process and communicates via Pipes.
# =====================================================================
def _worker_fn(conn, obelix_py, env_kwargs, stuck_limit, seed_offset):
    """Target function for each subprocess worker."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    OBELIX = mod.OBELIX

    stuck_counter = 0
    env = None

    while True:
        cmd, data = conn.recv()

        if cmd == "reset":
            if isinstance(data, dict):
                ep_seed = int(data["seed"])
                runtime_env_kwargs = dict(env_kwargs)
                runtime_env_kwargs.update(data.get("env_kwargs", {}))
            else:
                ep_seed = int(data)
                runtime_env_kwargs = env_kwargs
            stuck_counter = 0
            env = OBELIX(**runtime_env_kwargs, seed=ep_seed)
            obs = env.reset(seed=ep_seed)
            conn.send(np.array(obs, dtype=np.float32))

        elif cmd == "step":
            action_str = data
            obs2, reward, done = env.step(action_str, render=False)
            obs2 = np.array(obs2, dtype=np.float32)

            # Stuck detection inside worker
            if obs2[17] > 0.5:
                stuck_counter += 1
            else:
                stuck_counter = 0
            if stuck_counter >= stuck_limit:
                done = True
                reward -= 100.0
                stuck_counter = 0

            conn.send((obs2, float(reward), bool(done)))

        elif cmd == "close":
            conn.close()
            return


class SubprocVecEnv:
    """
    Manages N environment workers in separate processes.
    Provides batched reset() and step() that return (N, OBS_DIM) arrays.
    """
    def __init__(self, obelix_py, num_envs, env_kwargs, stuck_limit=20, seed=0):
        self.num_envs    = num_envs
        self.seed        = seed
        self.parent_conns, self.child_conns = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.processes = []

        for i in range(num_envs):
            p = mp.Process(
                target=_worker_fn,
                args=(self.child_conns[i], obelix_py, env_kwargs, stuck_limit, seed + i),
                daemon=True,
            )
            p.start()
            self.processes.append(p)

    def reset(self, episode: int = 0, env_kwargs_override: dict | None = None):
        for i, conn in enumerate(self.parent_conns):
            conn.send(
                (
                    "reset",
                    {
                        "seed": self.seed + episode * self.num_envs + i,
                        "env_kwargs": env_kwargs_override or {},
                    },
                )
            )
        return np.stack([conn.recv() for conn in self.parent_conns])  # (N, OBS_DIM)

    def step(self, actions):
        """actions: list of int, length = num_envs"""
        for conn, a in zip(self.parent_conns, actions):
            conn.send(("step", ACTIONS[a]))
        results = [conn.recv() for conn in self.parent_conns]
        obs2, rewards, dones = zip(*results)
        return (
            np.stack(obs2),            # (N, OBS_DIM)
            np.array(rewards),         # (N,)
            np.array(dones, dtype=bool)  # (N,)
        )

    def close(self):
        for conn in self.parent_conns:
            conn.send(("close", None))
        for p in self.processes:
            p.join(timeout=5)


# =====================================================================
# RND
# =====================================================================
class RNDTarget(nn.Module):
    def __init__(self, in_dim=OBS_DIM, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim,128), nn.LeakyReLU(0.1), nn.Linear(128,128), nn.LeakyReLU(0.1), nn.Linear(128,out_dim))
        for p in self.parameters(): p.requires_grad = False
    def forward(self, x): return self.net(x)

class RNDPredictor(nn.Module):
    def __init__(self, in_dim=OBS_DIM, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim,128), nn.LeakyReLU(0.1), nn.Linear(128,128), nn.LeakyReLU(0.1), nn.Linear(128,out_dim))
    def forward(self, x): return self.net(x)

class RND:
    def __init__(self, in_dim=OBS_DIM, out_dim=64, lr=1e-4, device="cpu", beta=0.1):
        self.target    = RNDTarget(in_dim, out_dim).to(device)
        self.predictor = RNDPredictor(in_dim, out_dim).to(device)
        self.opt       = optim.Adam(self.predictor.parameters(), lr=lr)
        self.device    = device
        self.beta      = beta
        self._mean, self._var, self._cnt = 0.0, 1.0, 1

    def intrinsic_reward(self, obs_t: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(): t = self.target(obs_t)
        p = self.predictor(obs_t)
        return F.mse_loss(p, t.detach(), reduction="none").mean(dim=1)

    def update(self, obs_t: torch.Tensor):
        t = self.target(obs_t).detach()
        p = self.predictor(obs_t)
        loss = F.mse_loss(p, t)
        self.opt.zero_grad(); loss.backward(); self.opt.step()

    def normalized(self, obs_t: torch.Tensor) -> torch.Tensor:
        raw = self.intrinsic_reward(obs_t)
        self._cnt += raw.numel()
        self._mean += (raw.mean().item() - self._mean) / self._cnt
        self._var   = max(self._var + ((raw - self._mean)**2).mean().item() / self._cnt, 1e-6)
        return self.beta * (raw - self._mean) / np.sqrt(self._var)


# =====================================================================
# SAC NETWORKS (Discrete)
# =====================================================================
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(OBS_DIM,256), nn.ReLU(), nn.Linear(256,256), nn.ReLU(), nn.Linear(256,N_ACTIONS))
    def forward(self, x):
        probs = F.softmax(self.net(x), dim=-1)
        return probs, F.log_softmax(self.net(x), dim=-1)
    def get_action(self, x):
        probs, log_p = self.forward(x)
        return torch.distributions.Categorical(probs).sample(), log_p, probs

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(OBS_DIM,256), nn.ReLU(), nn.Linear(256,256), nn.ReLU(), nn.Linear(256,N_ACTIONS))
    def forward(self, x): return self.net(x)


# =====================================================================
# REPLAY BUFFER
# =====================================================================
class ReplayBuffer:
    def __init__(self, cap=300_000):
        self.buf = collections.deque(maxlen=cap)
    def push(self, s, a, r, s2, d):
        self.buf.append((s, int(a), float(r), s2, float(d)))
    def sample(self, n):
        s,a,r,s2,d = zip(*random.sample(self.buf, n))
        return (torch.tensor(np.array(s),dtype=torch.float32),
                torch.tensor(a,dtype=torch.long),
                torch.tensor(r,dtype=torch.float32),
                torch.tensor(np.array(s2),dtype=torch.float32),
                torch.tensor(d,dtype=torch.float32))
    def __len__(self): return len(self.buf)


# =====================================================================
# DISCRETE SAC AGENT
# =====================================================================
class DiscreteSAC:
    def __init__(self, lr=3e-4, gamma=0.99, tau=0.005, device="cpu"):
        self.gamma, self.tau, self.device = gamma, tau, device
        self.actor   = Actor().to(device)
        self.critic1 = Critic().to(device)
        self.critic2 = Critic().to(device)
        self.tgt1    = Critic().to(device); self.tgt1.load_state_dict(self.critic1.state_dict())
        self.tgt2    = Critic().to(device); self.tgt2.load_state_dict(self.critic2.state_dict())

        # target_entropy_ratio: 0.5 = allow policy to be 50% of max entropy
        # 0.98 (old) was too high — even random policy couldn't meet it → alpha explosion
        self.target_entropy = -np.log(1.0 / N_ACTIONS) * 0.5
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha     = self.log_alpha.exp().item()

        self.a_opt  = optim.Adam(self.actor.parameters(), lr=lr)
        self.c_opt  = optim.Adam(list(self.critic1.parameters())+list(self.critic2.parameters()), lr=lr)
        self.al_opt = optim.Adam([self.log_alpha], lr=lr)

    @torch.no_grad()
    def select_actions(self, obs_batch: np.ndarray) -> np.ndarray:
        x = torch.tensor(obs_batch, dtype=torch.float32).to(self.device)
        probs, _ = self.actor(x)
        return torch.distributions.Categorical(probs).sample().cpu().numpy()

    def update(self, replay: ReplayBuffer, batch: int, rnd: RND):
        s, a, r, s2, d = [t.to(self.device) for t in replay.sample(batch)]
        r_int = rnd.normalized(s2); rnd.update(s2)
        r_total = r + r_int.detach()

        with torch.no_grad():
            p2, lp2 = self.actor(s2)
            minq = torch.min(self.tgt1(s2), self.tgt2(s2))
            v_next = (p2 * (minq - self.alpha * lp2)).sum(1)
            tq = r_total + self.gamma * (1-d) * v_next

        q1 = self.critic1(s).gather(1,a.unsqueeze(1)).squeeze()
        q2 = self.critic2(s).gather(1,a.unsqueeze(1)).squeeze()
        c_loss = F.mse_loss(q1,tq) + F.mse_loss(q2,tq)
        self.c_opt.zero_grad(); c_loss.backward(); self.c_opt.step()

        p, lp = self.actor(s)
        minq_s = torch.min(self.critic1(s), self.critic2(s))
        a_loss = (p * (self.alpha * lp - minq_s)).sum(1).mean()
        self.a_opt.zero_grad(); a_loss.backward(); self.a_opt.step()

        entropy = -(p * lp).sum(1).detach()
        al_loss = (self.log_alpha * (entropy - self.target_entropy)).mean()
        self.al_opt.zero_grad(); al_loss.backward(); self.al_opt.step()
        # Clamp to prevent runaway alpha (range: exp(-3)~0.05 to exp(0.5)~1.65)
        self.log_alpha.data.clamp_(-3.0, 0.5)
        self.alpha = self.log_alpha.exp().item()

        for tp, p in zip(self.tgt1.parameters(), self.critic1.parameters()):
            tp.data.mul_(1-self.tau).add_(self.tau*p.data)
        for tp, p in zip(self.tgt2.parameters(), self.critic2.parameters()):
            tp.data.mul_(1-self.tau).add_(self.tau*p.data)


# =====================================================================
# MAIN
# =====================================================================
def main():
    mp.set_start_method("spawn", force=True)

    ap = argparse.ArgumentParser(description="Vectorized SAC+RND for OBELIX")
    ap.add_argument("--obelix_py",        type=str,   required=True)
    ap.add_argument("--out",              type=str,   default="sac_vec.pth")
    ap.add_argument("--episodes",         type=int,   default=2000)
    ap.add_argument("--max_steps",        type=int,   default=1000)
    ap.add_argument("--num_envs",         type=int,   default=4,
                    help="Number of parallel environment workers. Match to CPU core count.")
    ap.add_argument("--difficulty",       type=int,   default=3)
    ap.add_argument("--wall_obstacles",   action="store_true")
    ap.add_argument("--box_speed",        type=int,   default=2)
    ap.add_argument("--scaling_factor",   type=int,   default=5)
    ap.add_argument("--arena_size",       type=int,   default=500)
    ap.add_argument("--gamma",            type=float, default=0.995)
    ap.add_argument("--lr",               type=float, default=1e-4)
    ap.add_argument("--batch",            type=int,   default=128,
                    help="Larger batch keeps GPU busy. Scales with num_envs.")
    ap.add_argument("--replay_start",     type=int,   default=12000)
    ap.add_argument("--updates_per_step", type=int,   default=1)
    ap.add_argument("--rnd_beta",         type=float, default=0.02)
    ap.add_argument("--rnd_beta_end",     type=float, default=0.0,
                    help="RND beta decays from rnd_beta to rnd_beta_end over training.")
    ap.add_argument("--load_weights",     type=str,   default="",
                    help="Path to pre-trained actor weights for curriculum learning.")
    ap.add_argument("--seed",             type=int,   default=0)
    ap.add_argument("--curriculum",       action="store_true",
                    help="Use a built-in curriculum that progresses through easier difficulties.")
    ap.add_argument("--curriculum_difficulties", type=str, default="0,2,3",
                    help="Comma-separated difficulty stages used when --curriculum is set.")
    ap.add_argument("--curriculum_ratios", type=str, default="0.25,0.35,0.40",
                    help="Comma-separated stage fractions used when --curriculum is set.")
    ap.set_defaults(curriculum=True, wall_obstacles=True)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Parallel envs: {args.num_envs}")

    if args.curriculum:
        curriculum_plan = _build_curriculum_plan(
            total_episodes=args.episodes,
            difficulties=_parse_int_csv(args.curriculum_difficulties),
            ratios=_parse_float_csv(args.curriculum_ratios),
        )
        print("Curriculum plan:")
        for idx, stage in enumerate(curriculum_plan, start=1):
            print(
                f"  Stage {idx}: episodes {stage['start'] + 1}-{stage['end']} "
                f"difficulty={stage['difficulty']} wall_obstacles={stage['wall_obstacles']}"
            )
    else:
        curriculum_plan = None

    env_kwargs = dict(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
    )

    vec_env = SubprocVecEnv(args.obelix_py, args.num_envs, env_kwargs, seed=args.seed)
    agent   = DiscreteSAC(lr=args.lr, gamma=args.gamma, device=device)

    # Load pre-trained actor weights for curriculum learning
    if args.load_weights:
        sd = torch.load(args.load_weights, map_location=device)
        agent.actor.load_state_dict(sd, strict=True)
        print(f"Loaded actor weights from {args.load_weights} (curriculum mode)")

    rnd     = RND(beta=args.rnd_beta, device=device)
    replay  = ReplayBuffer()

    # Episode-level tracking per worker
    ep_rets  = np.zeros(args.num_envs)
    ep_done  = np.zeros(args.num_envs, dtype=bool)
    n_done   = 0
    total_success = 0
    best_return   = -float("inf")
    last_logged   = -1  # prevents duplicate prints at the same n_done checkpoint

    last50_ret = []; last50_suc = []

    # Initial reset — episode 0 for all workers
    initial_stage = _curriculum_stage_for_episode(curriculum_plan, 0) if curriculum_plan else None
    initial_override = None if initial_stage is None else {
        "difficulty": initial_stage["difficulty"],
        "wall_obstacles": initial_stage["wall_obstacles"],
    }
    obs = vec_env.reset(episode=0, env_kwargs_override=initial_override)          # (N, OBS_DIM)
    ep_counts = np.zeros(args.num_envs, dtype=int)   # which episode each worker is on

    while n_done < args.episodes:
        # ---------- collect one step from ALL envs ----------
        if len(replay) < args.replay_start:
            actions = np.array([random.randint(0, N_ACTIONS-1) for _ in range(args.num_envs)])
        else:
            actions = agent.select_actions(obs)  # batched GPU call

        obs2, rewards, dones = vec_env.step(actions)

        for i in range(args.num_envs):
            replay.push(obs[i], actions[i], rewards[i], obs2[i], dones[i])
            ep_rets[i] += rewards[i]

            if dones[i]:
                success = int(rewards[i] > 1000.0)
                last50_ret.append(ep_rets[i]); last50_suc.append(success)
                if len(last50_ret) > 50: last50_ret.pop(0); last50_suc.pop(0)
                if ep_rets[i] > best_return:
                    best_return = ep_rets[i]
                    torch.save(agent.actor.state_dict(), args.out)
                total_success += success
                ep_rets[i] = 0
                n_done += 1
                ep_counts[i] += 1

                next_stage = _curriculum_stage_for_episode(curriculum_plan, n_done) if curriculum_plan else None
                next_override = {} if next_stage is None else {
                    "difficulty": next_stage["difficulty"],
                    "wall_obstacles": next_stage["wall_obstacles"],
                }
                vec_env.parent_conns[i].send(
                    (
                        "reset",
                        {
                            "seed": args.seed + n_done + i,
                            "env_kwargs": next_override,
                        },
                    )
                )
                obs2[i] = vec_env.parent_conns[i].recv()

        obs = obs2

        # ---------- GPU update ----------
        if len(replay) >= args.replay_start:
            for _ in range(args.updates_per_step):
                agent.update(replay, args.batch, rnd)

        if n_done > 0 and n_done % 50 == 0 and n_done != last_logged and len(last50_ret) > 0:
            last_logged = n_done
            current_stage = _curriculum_stage_for_episode(curriculum_plan, n_done) if curriculum_plan else None
            stage_msg = ""
            if current_stage is not None:
                stage_msg = (
                    f" stage_difficulty={current_stage['difficulty']}"
                    f" stage_walls={int(current_stage['wall_obstacles'])}"
                )
            print(
                f"Ep {n_done}/{args.episodes} "
                f"mean={np.mean(last50_ret):.1f} "
                f"max={np.max(last50_ret):.1f} "
                f"succ_50={int(np.sum(last50_suc))} "
                f"alpha={agent.alpha:.3f} "
                f"buf={len(replay)} "
                f"best={best_return:.1f}"
                f"{stage_msg}"
            )

    vec_env.close()
    print(f"\nDone! Successes: {total_success}/{args.episodes}")
    torch.save(agent.actor.state_dict(), args.out)
    print(f"Saved actor → {args.out}")


if __name__ == "__main__":
    main()
