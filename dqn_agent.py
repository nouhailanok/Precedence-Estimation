# """
# dqn_agent.py — Pure DQN baseline for CartPole-v1
# =================================================

# Design decisions (mirroring the PPO file's spirit):
#   - Separate online Q-network and frozen target Q-network (hard update every C steps)
#   - Experience replay buffer (uniform sampling)
#   - ε-greedy exploration with linear decay
#   - Separate Adam optimiser for the Q-network
#   - Per-episode tracking for a clean learning curve

# Output
# ------
#     DQN_plots/
#         dqn_curve.png
#         dqn_agent.pth
# """

# import os
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import gymnasium as gym
# from collections import deque

# # ─────────────────────────────────────────────
# # PATHS
# # ─────────────────────────────────────────────
# BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
# PLOTS_DIR = os.path.join(BASE_DIR, "DQN_plots")
# ENV_NAME  = "CartPole-v1"
# os.makedirs(PLOTS_DIR, exist_ok=True)

# # ─────────────────────────────────────────────
# # CONFIG
# # ─────────────────────────────────────────────
# N_EPISODES        = 600
# MAX_STEPS         = 500

# HIDDEN_SIZES      = (64, 64)

# BUFFER_SIZE       = 50_000
# BATCH_SIZE        = 64

# LR                = 1e-3
# GAMMA             = 0.99

# TARGET_UPDATE_FREQ = 200

# EPS_START         = 1.0
# EPS_END           = 0.01
# EPS_DECAY_STEPS   = 10_000

# WARMUP_STEPS      = 1_000

# SEED   = 42
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)


# # ─────────────────────────────────────────────
# # Q-NETWORK
# # ─────────────────────────────────────────────
# def build_mlp(in_dim: int, hidden_sizes: tuple, out_dim: int) -> nn.Sequential:
#     layers, prev = [], in_dim
#     for h in hidden_sizes:
#         layers += [nn.Linear(prev, h), nn.ReLU()]
#         prev = h
#     layers.append(nn.Linear(prev, out_dim))
#     return nn.Sequential(*layers)


# class QNetwork(nn.Module):
#     def __init__(self, obs_dim: int, n_actions: int,
#                  hidden: tuple = HIDDEN_SIZES):
#         super().__init__()
#         self.net = build_mlp(obs_dim, hidden, n_actions)

#     def forward(self, obs: torch.Tensor) -> torch.Tensor:
#         return self.net(obs)


# # ─────────────────────────────────────────────
# # REPLAY BUFFER
# # ─────────────────────────────────────────────
# class ReplayBuffer:
#     def __init__(self, capacity: int = BUFFER_SIZE):
#         self.buf = deque(maxlen=capacity)

#     def push(self, obs, action, reward, next_obs, done):
#         self.buf.append((
#             np.asarray(obs,      dtype=np.float32),
#             int(action),
#             float(reward),
#             np.asarray(next_obs, dtype=np.float32),
#             float(done),
#         ))

#     def sample(self, batch_size: int = BATCH_SIZE):
#         batch = random.sample(self.buf, batch_size)
#         obs, actions, rewards, next_obs, dones = zip(*batch)
#         return (
#             torch.tensor(np.stack(obs),      dtype=torch.float32, device=DEVICE),
#             torch.tensor(actions,            dtype=torch.int64,   device=DEVICE),
#             torch.tensor(rewards,            dtype=torch.float32, device=DEVICE),
#             torch.tensor(np.stack(next_obs), dtype=torch.float32, device=DEVICE),
#             torch.tensor(dones,              dtype=torch.float32, device=DEVICE),
#         )

#     def __len__(self):
#         return len(self.buf)


# # ─────────────────────────────────────────────
# # ε-GREEDY POLICY
# # ─────────────────────────────────────────────
# def get_epsilon(step: int) -> float:
#     frac = min(step / EPS_DECAY_STEPS, 1.0)
#     return EPS_START + frac * (EPS_END - EPS_START)


# @torch.no_grad()
# def select_action(obs: np.ndarray, q_net: QNetwork,
#                   epsilon: float, n_actions: int) -> int:
#     if random.random() < epsilon:
#         return random.randint(0, n_actions - 1)
#     obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
#     return int(q_net(obs_t).argmax(dim=1).item())


# # ─────────────────────────────────────────────
# # DQN UPDATE
# # ─────────────────────────────────────────────
# def dqn_update(q_net: QNetwork, target_net: QNetwork,
#                optimizer: optim.Optimizer,
#                replay: ReplayBuffer) -> float:
#     obs_b, act_b, rew_b, next_obs_b, done_b = replay.sample()

#     q_values = q_net(obs_b).gather(1, act_b.unsqueeze(1)).squeeze(1)

#     with torch.no_grad():
#         next_q    = target_net(next_obs_b).max(dim=1).values
#         td_target = rew_b + GAMMA * next_q * (1.0 - done_b)

#     loss = F.mse_loss(q_values, td_target)

#     optimizer.zero_grad()
#     loss.backward()
#     nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
#     optimizer.step()

#     return loss.item()


# # ─────────────────────────────────────────────
# # TRAINING LOOP
# # ─────────────────────────────────────────────
# def train_dqn():
#     env     = gym.make(ENV_NAME)
#     obs_dim = env.observation_space.shape[0]
#     n_act   = env.action_space.n

#     q_net      = QNetwork(obs_dim, n_act, HIDDEN_SIZES).to(DEVICE)
#     target_net = QNetwork(obs_dim, n_act, HIDDEN_SIZES).to(DEVICE)
#     target_net.load_state_dict(q_net.state_dict())
#     target_net.eval()

#     optimizer = optim.Adam(q_net.parameters(), lr=LR)
#     replay    = ReplayBuffer(BUFFER_SIZE)

#     print(f"[DQN] Pure baseline — no world model")
#     print(f"      N_EPISODES={N_EPISODES} | BUFFER={BUFFER_SIZE} | BATCH={BATCH_SIZE}")
#     print(f"      LR={LR} | GAMMA={GAMMA} | TARGET_UPDATE={TARGET_UPDATE_FREQ}")
#     print(f"      ε: {EPS_START} → {EPS_END} over {EPS_DECAY_STEPS} steps")

#     all_rewards, cumulative_mean = [], []
#     losses      = []
#     steps_total = 0

#     for episode in range(1, N_EPISODES + 1):
#         obs, _ = env.reset(seed=SEED + episode)
#         ep_return = 0.0

#         for _ in range(MAX_STEPS):
#             epsilon = get_epsilon(steps_total)
#             action  = select_action(obs, q_net, epsilon, n_act)

#             next_obs, reward, terminated, truncated, _ = env.step(action)
#             done = terminated or truncated

#             replay.push(obs, action, reward, next_obs, done)
#             obs          = next_obs
#             ep_return   += reward
#             steps_total += 1

#             if len(replay) >= WARMUP_STEPS:
#                 loss = dqn_update(q_net, target_net, optimizer, replay)
#                 losses.append(loss)

#             if steps_total % TARGET_UPDATE_FREQ == 0:
#                 target_net.load_state_dict(q_net.state_dict())

#             if done:
#                 break

#         all_rewards.append(ep_return)
#         cumulative_mean.append(np.mean(all_rewards))

#         if episode % 50 == 0:
#             print(f"Episode {episode:03d}/{N_EPISODES}  "
#                   f"Return={ep_return:.1f}  "
#                   f"MeanAll={cumulative_mean[-1]:.2f}  "
#                   f"ε={get_epsilon(steps_total):.3f}  "
#                   f"Steps={steps_total}")

#     env.close()

#     # ── Single reward plot ───────────────────────────────────────────────────
#     fig, ax = plt.subplots(figsize=(9, 5))

#     ax.plot(all_rewards,      alpha=0.35, label="Episode Return")
#     ax.plot(cumulative_mean, linewidth=2,  label="Mean (All Episodes)")

#     # Annotate the final mean reward as a value on the plot
#     final_mean = cumulative_mean[-1]
#     ax.axhline(y=final_mean, color='steelblue', linestyle='--', alpha=0.6, linewidth=1)
#     ax.text(len(cumulative_mean) - 1, final_mean + 5,
#             f"Final Mean (All): {final_mean:.1f}",
#             ha='right', va='bottom', color='steelblue', fontsize=10)

#     ax.grid(alpha=0.3)
#     ax.set_xlabel("Episode")
#     ax.set_ylabel("Return")
#     ax.set_title("DQN — CartPole-v1 (no Precedence estimation)")
#     ax.legend()

#     plt.tight_layout()
#     fig_path = os.path.join(PLOTS_DIR, "dqn_curve.png")
#     plt.savefig(fig_path, dpi=150)
#     plt.show()
#     print(f"[Plot] Saved → {fig_path}")

#     # ── Save weights ─────────────────────────────────────────────────────────
#     pth_path = os.path.join(PLOTS_DIR, "dqn_agent.pth")
#     torch.save({
#         "q_net":      q_net.state_dict(),
#         "target_net": target_net.state_dict(),
#     }, pth_path)
#     print(f"[Model] Saved → {pth_path}")

#     # ── Greedy evaluation ────────────────────────────────
#     eval_rewards = []
#     eval_env     = gym.make(ENV_NAME)
#     q_net.eval()

#     for i in range(10):   # 10 greedy episodes → stable mean
#         obs_e, _ = eval_env.reset(seed=SEED + 200 + i)
#         ep_r, done_e = 0.0, False
#         while not done_e:
#             with torch.no_grad():
#                 obs_t = torch.tensor(obs_e, dtype=torch.float32, device=DEVICE).unsqueeze(0)
#                 a_e   = int(q_net(obs_t).argmax(dim=1).item())
#             obs_e, r_e, term_e, trunc_e, _ = eval_env.step(a_e)
#             done_e = term_e or trunc_e
#             ep_r  += r_e
#         eval_rewards.append(ep_r)

#     eval_env.close()
#     print(f"[Eval] Greedy mean reward (10 episodes) = {np.mean(eval_rewards):.2f}  "
#           f"(min={min(eval_rewards):.0f}, max={max(eval_rewards):.0f})")


# # ─────────────────────────────────────────────
# # ENTRY POINT
# # ─────────────────────────────────────────────
# if __name__ == "__main__":
#     train_dqn()

"""
dqn_agent.py — Pure DQN baseline
=================================
Supports CartPole-v1, LunarLander-v3, and MountainCar-v0.
No world model, no synthetic transitions.

Usage
-----
    python dqn_agent.py                          # default: CartPole-v1
    python dqn_agent.py --env LunarLander-v3
    python dqn_agent.py --env MountainCar-v0

Design decisions:
  - Separate online Q-network and frozen target Q-network (hard update every C steps)
  - Experience replay buffer (uniform sampling)
  - ε-greedy exploration with linear decay
  - Per-environment hyperparameter profiles (reward scale, episode length,
    exploration budget, and network capacity all differ across envs)
  - Reward shaping for MountainCar-v0 (sparse reward makes raw DQN impractical)

Output
------
    DQN_plots/<env>/
        dqn_curve.png
        dqn_agent.pth

Why per-env tuning is necessary
--------------------------------
CartPole-v1   : dense +1/step, short episodes (≤500), 4-dim state, 2 actions.
                Fast to learn — small net and buffer are fine.

LunarLander-v3: shaped reward (±100 land/crash, −0.3/step, leg contacts),
                episodes up to 1000 steps, 8-dim state, 4 actions.
                Needs a larger net, more buffer, and more episodes to converge.

MountainCar-v0: reward = −1/step until goal (pos ≥ 0.5), max 200 steps.
                Purely sparse signal → almost no gradient without shaping.
                We add a potential-based shaping term: Δ(position + speed²)
                which is reward-shaping theory-safe (preserves optimal policy).
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
    },

    "LunarLander-v3": {
        # Needs more capacity and experience than CartPole.
        # The shaped reward signal is already dense enough.
        # ── training ──
        "n_episodes":        800,
        "max_steps":         1000,
        # ── network ──
        "hidden_sizes":      (128, 128),   # larger: 8-dim state, 4 actions
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
        "eps_decay_steps":   50_000,       # slower decay — harder env
        # ── grad clipping ──
        "max_grad_norm":     10.0,
        # ── plotting ──
        "solve_threshold":   200,          # gym considers ≥200 mean-100 solved
        "loss_smooth_window": 500,
        # ── reward shaping ──
        "shaped_reward":     False,
    },

    "MountainCar-v0": {
        # Sparse −1/step reward: almost no gradient without shaping.
        # shaped_reward=True adds a potential-based bonus that preserves
        # the optimal policy (Ng et al. 1999).
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
        "eps_decay_steps":   30_000,       # slower — needs long exploration
        # ── grad clipping ──
        "max_grad_norm":     10.0,
        # ── plotting ──
        "solve_threshold":   -110,         # MountainCar: higher (less negative) is better
        "loss_smooth_window": 300,
        # ── reward shaping ──
        "shaped_reward":     True,
    },
}

SUPPORTED_ENVS = list(ENV_PROFILES.keys())


# ─────────────────────────────────────────────
# REWARD SHAPING
# ─────────────────────────────────────────────
def shape_reward(obs, next_obs, raw_reward, env_name: str) -> float:
    """
    Potential-based reward shaping  F(s, s') = γ·Φ(s') − Φ(s).
    Only active for MountainCar-v0 where the raw signal is too sparse.

    Φ(s) = position + velocity²
      - position encourages moving right toward the goal
      - velocity² encourages building kinetic energy (swinging)
    """
    if env_name != "MountainCar-v0":
        return raw_reward
    gamma   = ENV_PROFILES["MountainCar-v0"]["gamma"]
    phi_s   = obs[0]      + obs[1] ** 2 # potential at current state
    phi_s2  = next_obs[0] + next_obs[1] ** 2 # potential at next state
    bonus   = gamma * phi_s2 - phi_s
    return raw_reward + bonus


# ─────────────────────────────────────────────
# Q-NETWORK
# ─────────────────────────────────────────────
def build_mlp(in_dim: int, hidden_sizes: tuple, out_dim: int) -> nn.Sequential:
    """ReLU MLP — standard for Q-networks (avoids saturation on unbounded Q-values)."""
    layers, prev = [], in_dim
    for h in hidden_sizes:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """
    Outputs Q(s, a) for all actions simultaneously.
    Shape: (batch, n_actions)
    """
    def __init__(self, obs_dim: int, n_actions: int, hidden: tuple):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ─────────────────────────────────────────────
# REPLAY BUFFER
# ─────────────────────────────────────────────
class ReplayBuffer:
    """Uniform experience replay — circular deque of (s, a, r, s', done)."""
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
    """Linear decay from eps_start → eps_end over eps_decay_steps."""
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
    """
    One gradient step on the Bellman TD loss:
        L = MSE( r + γ · max_a' Q_target(s', a') · (1−done),  Q_online(s, a) )
    """
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
    obs_dim = env.observation_space.shape[0]
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
    if cfg["shaped_reward"]:
        print(f"  reward shaping: ON (potential-based, γ·Φ(s')−Φ(s))")
    print(f"{'='*60}\n")

    all_rewards, mean_rewards = [], []
    losses      = []
    steps_total = 0

    for episode in range(1, cfg["n_episodes"] + 1):
        obs, _ = env.reset(seed=SEED + episode)
        ep_return = 0.0

        for _ in range(cfg["max_steps"]):
            epsilon = get_epsilon(steps_total, cfg)
            action  = select_action(obs, q_net, epsilon, n_act)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Apply reward shaping if configured for this env
            stored_reward = shape_reward(obs, next_obs, reward, env_name)

            replay.push(obs, action, stored_reward, next_obs, done)
            obs          = next_obs
            ep_return   += reward        # always track the RAW return for plotting
            steps_total += 1

            # Learn once buffer is warm
            if len(replay) >= cfg["warmup_steps"]:
                loss = dqn_update(q_net, target_net, optimizer, replay, cfg)
                losses.append(loss)

            # Hard-copy online → target
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

    # Learning curve
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

    # TD loss
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

    # Config annotation
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
    }, pth_path)
    print(f"[Model] Saved → {pth_path}")

    # ── Greedy evaluation ─────────────────────────────────────────────────────
    eval_env = gym.make(env_name)
    obs_e, _ = eval_env.reset(seed=SEED + 111)
    total_r  = 0.0
    done_e   = False
    q_net.eval()
    while not done_e:
        with torch.no_grad():
            obs_t = torch.tensor(obs_e, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            a_e   = int(q_net(obs_t).argmax(dim=1).item())
        obs_e, r_e, term_e, trunc_e, _ = eval_env.step(a_e)
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