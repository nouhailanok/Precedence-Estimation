"""
dqn_agent.py — Pure DQN baseline
=================================
Supports CartPole-v1, LunarLander-v3, MountainCar-v0, and CliffWalking-v1.
No world model, no synthetic transitions.

Usage
-----
    python dqn_agent.py                          # default: CartPole-v1
    python dqn_agent.py --env LunarLander-v3
    python dqn_agent.py --env MountainCar-v0
    python dqn_agent.py --env CliffWalking-v1

Design decisions:
  - Separate online Q-network and frozen target Q-network (hard update every C steps)
  - Experience replay buffer (uniform sampling)
  - ε-greedy exploration with linear decay
  - Per-environment hyperparameter profiles (reward scale, episode length,
    exploration budget, and network capacity all differ across envs)
  - Reward shaping for MountainCar-v0 (sparse reward makes raw DQN impractical)
  - One-hot state preprocessing wrapper for discrete gridworld environments

Output
------
    DQN_plots/<env>/
        dqn_curve.png
        dqn_agent.pth
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gymnasium as gym
from collections import deque

# ─────────────────────────────────────────────
# PATHS & SEED
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEED     = 42
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ─────────────────────────────────────────────
# PER-ENVIRONMENT HYPERPARAMETER PROFILES
# ─────────────────────────────────────────────
ENV_PROFILES = {
    "CartPole-v1": {
        # ── training ──
        "n_episodes":        600,
        "max_steps":         500,
        # ── network ──
        "hidden_sizes":      (64, 64),
        # ── replay ──
        "buffer_size":       50_000,
        "batch_size":        64,
        "warmup_steps":      1_000,
        # ── optimisation ──
        "lr":                1e-3,
        "gamma":             0.99,
        # ── target network ──
        "target_update_freq": 200,
        # ── exploration ──
        "eps_start":         1.0,
        "eps_end":           0.01,
        "eps_decay_steps":   10_000,
        # ── grad clipping ──
        "max_grad_norm":     10.0,
        # ── plotting ──
        "solve_threshold":   475,
        "loss_smooth_window": 200,
        # ── reward shaping ──
        "shaped_reward":     False,
        # ── observation ──
        "discrete_obs":      False,
        "n_states":          None,
    },

    "LunarLander-v3": {
        # ── training ──
        "n_episodes":        800,
        "max_steps":         1000,
        # ── network ──
        "hidden_sizes":      (128, 128),
        # ── replay ──
        "buffer_size":       100_000,
        "batch_size":        128,
        "warmup_steps":      2_000,
        # ── optimisation ──
        "lr":                5e-4,
        "gamma":             0.99,
        # ── target network ──
        "target_update_freq": 500,
        # ── exploration ──
        "eps_start":         1.0,
        "eps_end":           0.05,
        "eps_decay_steps":   50_000,
        # ── grad clipping ──
        "max_grad_norm":     10.0,
        # ── plotting ──
        "solve_threshold":   200,
        "loss_smooth_window": 500,
        # ── reward shaping ──
        "shaped_reward":     False,
        # ── observation ──
        "discrete_obs":      False,
        "n_states":          None,
    },

    "MountainCar-v0": {
        # ── training ──
        "n_episodes":        1000,
        "max_steps":         200,
        # ── network ──
        "hidden_sizes":      (64, 64),
        # ── replay ──
        "buffer_size":       50_000,
        "batch_size":        64,
        "warmup_steps":      1_000,
        # ── optimisation ──
        "lr":                1e-3,
        "gamma":             0.99,
        # ── target network ──
        "target_update_freq": 300,
        # ── exploration ──
        "eps_start":         1.0,
        "eps_end":           0.01,
        "eps_decay_steps":   30_000,
        # ── grad clipping ──
        "max_grad_norm":     10.0,
        # ── plotting ──
        "solve_threshold":   -110,
        "loss_smooth_window": 300,
        # ── reward shaping ──
        "shaped_reward":     True,
        # ── observation ──
        "discrete_obs":      False,
        "n_states":          None,
    },

    "CliffWalking-v1": {
        # 4×12 grid = 48 discrete states.
        # Deep Q-learning maps continuous outputs, so states are one-hot encoded.
        # High target update frequency allows Q-estimates to stabilize.
        # ── training ──
        "n_episodes":        500,
        "max_steps":         300,
        # ── network ──
        "hidden_sizes":      (64, 64),
        # ── replay ──
        "buffer_size":       20_000,
        "batch_size":        64,
        "warmup_steps":      500,
        # ── optimisation ──
        "lr":                1e-3,
        "gamma":             0.99,
        # ── target network ──
        "target_update_freq": 200,
        # ── exploration ──
        "eps_start":         1.0,
        "eps_end":           0.01,
        "eps_decay_steps":   5_000,        # Fast decay — environment is highly directional
        # ── grad clipping ──
        "max_grad_norm":     1.0,          # Tighter gradient bounds due to sharp -100 jumps
        # ── plotting ──
        "solve_threshold":   -20,          # Optimal non-cliff trajectory score is -13
        "loss_smooth_window": 100,
        # ── reward shaping ──
        "shaped_reward":     False,        # Step costs are already inherently dense
        # ── observation ──
        "discrete_obs":      True,
        "n_states":          48,
    },
}

SUPPORTED_ENVS = list(ENV_PROFILES.keys())


# ─────────────────────────────────────────────
# OBSERVATION PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_obs(obs, cfg: dict) -> np.ndarray:
    """
    Transforms raw observations to a consistent float32 format.
    One-hot encodes discrete integer spaces into orthogonal coordinate features.
    """
    if cfg["discrete_obs"]:
        # Unwrap observation if wrapped inside numpy wrappers or tuples
        raw_idx = int(obs[0]) if isinstance(obs, (tuple, list, np.ndarray)) else int(obs)
        vec = np.zeros(cfg["n_states"], dtype=np.float32)
        vec[raw_idx] = 1.0
        return vec
    return np.asarray(obs, dtype=np.float32)


def get_obs_dim(cfg: dict, env: gym.Env) -> int:
    """Determines layer feature size for the network initialization."""
    if cfg["discrete_obs"]:
        return cfg["n_states"]
    return env.observation_space.shape[0]


# ─────────────────────────────────────────────
# REWARD SHAPING
# ─────────────────────────────────────────────
def shape_reward(obs, next_obs, raw_reward, env_name: str) -> float:
    """Potential-based reward shaping active only for MountainCar-v0."""
    if env_name != "MountainCar-v0":
        return raw_reward
    gamma   = ENV_PROFILES["MountainCar-v0"]["gamma"]
    phi_s   = obs[0]      + obs[1] ** 2
    phi_s2  = next_obs[0] + next_obs[1] ** 2
    bonus   = gamma * phi_s2 - phi_s
    return raw_reward + bonus


# ─────────────────────────────────────────────
# Q-NETWORK
# ─────────────────────────────────────────────
def build_mlp(in_dim: int, hidden_sizes: tuple, out_dim: int) -> nn.Sequential:
    """ReLU MLP — standard for Q-networks."""
    layers, prev = [], in_dim
    for h in hidden_sizes:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: tuple):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ─────────────────────────────────────────────
# REPLAY BUFFER
# ─────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buf.append((
            np.asarray(obs,      dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_obs, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int, device):
        batch = random.sample(self.buf, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            torch.tensor(np.stack(obs),      dtype=torch.float32, device=device),
            torch.tensor(actions,            dtype=torch.int64,   device=device),
            torch.tensor(rewards,            dtype=torch.float32, device=device),
            torch.tensor(np.stack(next_obs), dtype=torch.float32, device=device),
            torch.tensor(dones,              dtype=torch.float32, device=device),
        )

    def __len__(self):
        return len(self.buf)


# ─────────────────────────────────────────────
# ε-GREEDY POLICY
# ─────────────────────────────────────────────
def get_epsilon(step: int, cfg: dict) -> float:
    frac = min(step / cfg["eps_decay_steps"], 1.0)
    return cfg["eps_start"] + frac * (cfg["eps_end"] - cfg["eps_start"])


@torch.no_grad()
def select_action(obs: np.ndarray, q_net: QNetwork,
                  epsilon: float, n_actions: int) -> int:
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    return int(q_net(obs_t).argmax(dim=1).item())


# ─────────────────────────────────────────────
# DQN UPDATE
# ─────────────────────────────────────────────
def dqn_update(q_net: QNetwork, target_net: QNetwork,
               optimizer: optim.Optimizer,
               replay: ReplayBuffer, cfg: dict) -> float:
    obs_b, act_b, rew_b, next_obs_b, done_b = replay.sample(cfg["batch_size"], DEVICE)

    q_values = q_net(obs_b).gather(1, act_b.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q    = target_net(next_obs_b).max(dim=1).values
        td_target = rew_b + cfg["gamma"] * next_q * (1.0 - done_b)

    loss = F.mse_loss(q_values, td_target)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=cfg["max_grad_norm"])
    optimizer.step()

    return loss.item()


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
def train_dqn(env_name: str = "CartPole-v1"):
    assert env_name in SUPPORTED_ENVS, \
        f"Unsupported env '{env_name}'. Choose from: {SUPPORTED_ENVS}"

    cfg      = ENV_PROFILES[env_name]
    plots_dir = os.path.join(BASE_DIR, "DQN_plots", env_name)
    os.makedirs(plots_dir, exist_ok=True)

    env     = gym.make(env_name)
    obs_dim = get_obs_dim(cfg, env)
    n_act   = env.action_space.n

    q_net      = QNetwork(obs_dim, n_act, cfg["hidden_sizes"]).to(DEVICE)
    target_net = QNetwork(obs_dim, n_act, cfg["hidden_sizes"]).to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=cfg["lr"])
    replay    = ReplayBuffer(cfg["buffer_size"])

    print(f"\n{'='*60}")
    print(f"  DQN — {env_name}  (no world model)")
    print(f"  obs_dim={obs_dim} | n_actions={n_act} | device={DEVICE}")
    print(f"  hidden={cfg['hidden_sizes']} | lr={cfg['lr']} | γ={cfg['gamma']}")
    print(f"  buffer={cfg['buffer_size']} | batch={cfg['batch_size']} | warmup={cfg['warmup_steps']}")
    print(f"  ε: {cfg['eps_start']} → {cfg['eps_end']} over {cfg['eps_decay_steps']} steps")
    print(f"  target_update={cfg['target_update_freq']} | episodes={cfg['n_episodes']}")
    if cfg["discrete_obs"]:
        print(f"  obs encoding: one-hot ({cfg['n_states']} states)")
    if cfg["shaped_reward"]:
        print(f"  reward shaping: ON (potential-based, γ·Φ(s')−Φ(s))")
    print(f"{'='*60}\n")

    all_rewards, mean_rewards = [], []
    losses      = []
    steps_total = 0

    for episode in range(1, cfg["n_episodes"] + 1):
        raw_obs, _ = env.reset(seed=SEED + episode)
        obs = preprocess_obs(raw_obs, cfg)
        ep_return = 0.0

        for _ in range(cfg["max_steps"]):
            epsilon = get_epsilon(steps_total, cfg)
            action  = select_action(obs, q_net, epsilon, n_act)

            raw_next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_obs = preprocess_obs(raw_next_obs, cfg)
            stored_reward = shape_reward(obs, next_obs, reward, env_name)

            replay.push(obs, action, stored_reward, next_obs, done)
            obs          = next_obs
            ep_return   += reward
            steps_total += 1

            if len(replay) >= cfg["warmup_steps"]:
                loss = dqn_update(q_net, target_net, optimizer, replay, cfg)
                losses.append(loss)

            if steps_total % cfg["target_update_freq"] == 0:
                target_net.load_state_dict(q_net.state_dict())

            if done:
                break

        all_rewards.append(ep_return)
        mean_rewards.append(np.mean(all_rewards[-25:]))

        if episode % 50 == 0:
            print(f"Episode {episode:04d}/{cfg['n_episodes']}  "
                  f"Return={ep_return:8.1f}  "
                  f"Mean25={mean_rewards[-1]:8.2f}  "
                  f"ε={get_epsilon(steps_total, cfg):.3f}  "
                  f"Steps={steps_total}")

    env.close()

    # ── Plots ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"DQN — {env_name} (no world model)", fontsize=13)

    ax = axes[0]
    ax.plot(all_rewards,  alpha=0.30, linewidth=0.8, label="Episode Return")
    ax.plot(mean_rewards, linewidth=2.0, label="Mean-25")
    threshold = cfg["solve_threshold"]
    ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.45, linewidth=1,
               label=f"Threshold ({threshold})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return (raw)")
    ax.set_title("Learning Curve")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    w = cfg["loss_smooth_window"]
    if len(losses) >= w:
        smooth = np.convolve(losses, np.ones(w) / w, mode='valid')
        ax.plot(smooth, linewidth=1.5, color='#D85A30', label=f"Smoothed (w={w})")
    elif losses:
        ax.plot(losses, linewidth=0.8, alpha=0.6, color='#D85A30', label="TD Loss")
    ax.set_xlabel("Update step")
    ax.set_ylabel("TD Loss (MSE)")
    ax.set_title("TD Loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    shaped_str = "ON" if cfg["shaped_reward"] else "OFF"
    fig.text(0.5, -0.04,
             f"hidden={cfg['hidden_sizes']}  lr={cfg['lr']}  γ={cfg['gamma']}  "
             f"buffer={cfg['buffer_size']}  batch={cfg['batch_size']}  "
             f"target_update={cfg['target_update_freq']}  "
             f"ε_decay={cfg['eps_decay_steps']}  shaping={shaped_str}",
             ha='center', fontsize=7.5, color='#555555')

    plt.tight_layout()
    fig_path = os.path.join(plots_dir, "dqn_curve.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[Plot] Saved → {fig_path}")

    # ── Save weights ──────────────────────────────────────────────────────────
    pth_path = os.path.join(plots_dir, "dqn_agent.pth")
    torch.save({
        "q_net":      q_net.state_dict(),
        "target_net": target_net.state_dict(),
        "env_name":   env_name,
        "obs_dim":    obs_dim,
        "n_actions":  n_act,
        "hidden_sizes": cfg["hidden_sizes"],
        "discrete_obs": cfg["discrete_obs"],
        "n_states":     cfg["n_states"],
    }, pth_path)
    print(f"[Model] Saved → {pth_path}")

    # ── Greedy evaluation ─────────────────────────────────────────────────────
    eval_env = gym.make(env_name)
    raw_obs_e, _ = eval_env.reset(seed=SEED + 111)
    obs_e = preprocess_obs(raw_obs_e, cfg)
    total_r  = 0.0
    done_e   = False
    q_net.eval()
    while not done_e:
        with torch.no_grad():
            obs_t = torch.tensor(obs_e, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            a_e   = int(q_net(obs_t).argmax(dim=1).item())
        raw_obs_e, r_e, term_e, trunc_e, _ = eval_env.step(a_e)
        obs_e   = preprocess_obs(raw_obs_e, cfg)
        done_e  = term_e or trunc_e
        total_r += r_e
    print(f"[Eval] Greedy episode reward = {total_r}")
    eval_env.close()

    return q_net, all_rewards, mean_rewards


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN baseline — multi-env")
    parser.add_argument(
        "--env",
        default="CartPole-v1",
        choices=SUPPORTED_ENVS,
        help=f"Environment to train on. Choices: {SUPPORTED_ENVS}",
    )
    args = parser.parse_args()
    train_dqn(env_name=args.env)