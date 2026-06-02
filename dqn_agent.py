"""
dqn_agent.py — Pure DQN baseline
=================================
Supports CartPole-v1 and CliffWalking-v1 (no world model).

Usage
-----
    python dqn_agent.py                          # default: CartPole-v1
    python dqn_agent.py --env CliffWalking-v1

Metrics (train + greedy eval, alignés avec dqn_with_precedence_CartPole_full.ipynb)
------------------------------------------------------------------------------------
  Train : avg reward, std, AUC, n_episodes, n_steps
  Eval  : greedy sur EVAL_N_EPISODES — mean, std, min, max, median, success_rate, lengths

Output
------
    DQN_plots/<env>/
        dqn_curve.png
        dqn_metrics.json
        dqn_agent.pth
"""

import os
import json
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

# Greedy eval — même protocole que dqn_with_precedence_CartPole_full.ipynb
EVAL_N_EPISODES = 30
EVAL_SEED_OFFSET = 1000

def json_converter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


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
        # ── plotting / eval (500 = seuil DQN full notebook) ──
        "solve_threshold":   500,
        "loss_smooth_window": 200,
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
# METRICS & GREEDY EVAL  (compatible DQN full notebook)
# ─────────────────────────────────────────────
def learning_curve_auc(rewards: np.ndarray) -> float:
    """Aire sous la courbe des returns (trapèzes sur numéro d'épisode)."""
    if len(rewards) == 0:
        return 0.0
    if len(rewards) == 1:
        return float(rewards[0])
    return float(np.trapz(rewards.astype(np.float64), np.arange(len(rewards))))


def compute_training_metrics(all_rewards: list, mean_rewards: list,
                             n_episodes: int, n_steps: int) -> dict:
    arr = np.array(all_rewards, dtype=np.float32)
    last_n = min(50, len(arr))
    return {
        "train_avg_reward": float(arr.mean()) if len(arr) else 0.0,
        "train_std_reward": float(arr.std()) if len(arr) else 0.0,
        "train_final_mean": float(arr[-last_n:].mean()) if last_n else 0.0,
        "train_final_std": float(arr[-last_n:].std()) if last_n > 1 else 0.0,
        "train_auc": learning_curve_auc(arr),
        "train_mean25_final": float(mean_rewards[-1]) if mean_rewards else 0.0,
        "n_episodes": int(n_episodes),
        "n_steps": int(n_steps),
    }


def print_metrics_summary(train_m: dict, eval_m: dict, log_print=print):
    log_print(f"\n{'='*72}")
    log_print("  TRAINING METRICS")
    log_print(f"{'='*72}")
    log_print(f"  Episodes        : {train_m['n_episodes']}")
    log_print(f"  Steps (total)   : {train_m['n_steps']}")
    log_print(f"  Avg reward      : {train_m['train_avg_reward']:.2f}")
    log_print(f"  Std reward      : {train_m['train_std_reward']:.2f}")
    log_print(f"  AUC (learning)  : {train_m['train_auc']:.1f}")
    log_print(f"  Final mean (50) : {train_m['train_final_mean']:.2f} ± {train_m['train_final_std']:.2f}")

    log_print(f"\n{'='*72}")
    log_print(f"  GREEDY EVAL ({eval_m['n_episodes']} episodes)")
    log_print(f"{'='*72}")
    log_print(f"  Mean reward     : {eval_m['eval_mean']:.2f} ± {eval_m['eval_std']:.2f}")
    log_print(f"  Min / Median / Max : {eval_m['eval_min']:.0f} / "
              f"{eval_m['eval_median']:.0f} / {eval_m['eval_max']:.0f}")
    log_print(f"  Success rate    : {100*eval_m['success_rate']:.1f}% "
              f"({eval_m['n_solved']}/{eval_m['n_episodes']}) "
              f"[≥ {eval_m['solve_threshold']:.0f}]")
    log_print(f"  Mean ep length  : {eval_m['eval_length_mean']:.1f} ± "
              f"{eval_m['eval_length_std']:.1f}")
    log_print(f"  Eval steps      : {eval_m['total_steps']}")


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


@torch.no_grad()
def evaluate_dqn_greedy(q_net: QNetwork, env_name: str, cfg: dict,
                        n_episodes: int = EVAL_N_EPISODES,
                        seed_offset: int = EVAL_SEED_OFFSET) -> dict:
    """
    Évaluation greedy (ε=0) — même structure que evaluate_agent_greedy
    dans dqn_with_precedence_CartPole_full.ipynb.
    """
    env = gym.make(env_name)
    max_steps = cfg["max_steps"]
    solve_threshold = cfg["solve_threshold"]
    n_act = env.action_space.n

    was_training = q_net.training
    q_net.eval()

    rewards, lengths = [], []
    total_steps = 0

    for ep in range(n_episodes):
        raw_obs, _ = env.reset(seed=SEED + seed_offset + ep)
        obs = preprocess_obs(raw_obs, cfg)
        ep_reward, ep_len = 0.0, 0

        for _ in range(max_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            action = int(q_net(obs_t).argmax(dim=1).item())

            raw_next, reward, terminated, truncated, _ = env.step(action)
            obs = preprocess_obs(raw_next, cfg)
            ep_reward += reward
            ep_len += 1
            total_steps += 1
            if terminated or truncated:
                break

        rewards.append(ep_reward)
        lengths.append(ep_len)

    env.close()
    if was_training:
        q_net.train()

    rewards_arr = np.array(rewards, dtype=np.float32)
    lengths_arr = np.array(lengths, dtype=np.float32)

    return {
        "eval_rewards": [float(r) for r in rewards],
        "eval_lengths": [int(l) for l in lengths],
        "eval_mean": float(rewards_arr.mean()),
        "eval_std": float(rewards_arr.std()),
        "eval_min": float(rewards_arr.min()),
        "eval_max": float(rewards_arr.max()),
        "eval_median": float(np.median(rewards_arr)),
        "eval_length_mean": float(lengths_arr.mean()),
        "eval_length_std": float(lengths_arr.std()),
        "success_rate": float((rewards_arr >= solve_threshold).mean()),
        "n_solved": int((rewards_arr >= solve_threshold).sum()),
        "n_episodes": n_episodes,
        "total_steps": total_steps,
        "solve_threshold": float(solve_threshold),
        "n_actions": n_act,
    }


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

            replay.push(obs, action, reward, next_obs, done)
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

    n_episodes_done = len(all_rewards)
    train_metrics = compute_training_metrics(
        all_rewards, mean_rewards, n_episodes_done, steps_total
    )
    print(f"\n  [Eval] Greedy evaluation ({EVAL_N_EPISODES} episodes)...")
    eval_metrics = evaluate_dqn_greedy(q_net, env_name, cfg)
    print_metrics_summary(train_metrics, eval_metrics)

    metrics = {"env_name": env_name, "train": train_metrics, "eval": eval_metrics}
    metrics_path = os.path.join(plots_dir, "dqn_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=json_converter)
    print(f"[Metrics] Saved → {metrics_path}")

    # ── Plots ────────────────────────────────────────────────────────────────
    threshold = cfg["solve_threshold"]
    train_avg = train_metrics["train_avg_reward"]
    eval_mean = eval_metrics["eval_mean"]
    eval_std  = eval_metrics["eval_std"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"DQN — {env_name} (no world model)", fontsize=13)

    episodes_x = np.arange(1, len(all_rewards) + 1)

    ax = axes[0, 0]
    ax.plot(all_rewards, alpha=0.30, linewidth=0.8, label="Episode Return")
    ax.plot(mean_rewards, linewidth=2.0, label="Mean-25")
    ax.axhline(train_avg, color='#6340A0', linestyle='-', alpha=0.7, linewidth=1.5,
               label=f"Avg reward ({train_avg:.1f})")
    ax.axhline(eval_mean, color='#1D9E75', linestyle='--', alpha=0.85, linewidth=1.5,
               label=f"Greedy eval ({eval_mean:.1f}±{eval_std:.1f})")
    ax.axhline(threshold, color='r', linestyle=':', alpha=0.5,
               label=f"Solved (≥{threshold:.0f})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return (raw)")
    ax.set_title("Learning Curve + Avg / Greedy Eval")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    window = 25
    if len(all_rewards) >= window:
        smooth = np.convolve(all_rewards, np.ones(window) / window, mode='valid')
        ax.plot(episodes_x[window - 1:], smooth, linewidth=2.0, color='#1D9E75',
                label=f"Smoothed (w={window})")
    ax.axhline(train_avg, color='#6340A0', linestyle='-', alpha=0.7,
               label=f"Avg ({train_avg:.1f})")
    ax.axhline(eval_mean, color='#0F6E56', linestyle='--', alpha=0.85,
               label=f"Greedy eval ({eval_mean:.1f})")
    ax.axhline(threshold, color='r', linestyle=':', alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(f"Smoothed (w={window})")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.bar(["Train\n(avg)", "Train\n(final 50)", f"Greedy\n({EVAL_N_EPISODES} eps)"],
           [train_metrics["train_avg_reward"],
            train_metrics["train_final_mean"],
            eval_metrics["eval_mean"]],
           yerr=[0, train_metrics["train_final_std"], eval_metrics["eval_std"]],
           capsize=5, color=['#6340A0', '#878787', '#1D9E75'])
    ax.axhline(threshold, color='r', linestyle='--', alpha=0.45, label=f"Solved ({threshold})")
    ax.set_ylabel("Reward")
    ax.set_title("Avg Reward Comparison")
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1, 1]
    w = cfg["loss_smooth_window"]
    if len(losses) >= w:
        smooth_loss = np.convolve(losses, np.ones(w) / w, mode='valid')
        ax.plot(smooth_loss, linewidth=1.5, color='#D85A30', label=f"TD Loss (w={w})")
    elif losses:
        ax.plot(losses, linewidth=0.8, alpha=0.6, color='#D85A30', label="TD Loss")
    ax.set_xlabel("Update step")
    ax.set_ylabel("TD Loss (MSE)")
    ax.set_title("TD Loss")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.text(0.5, 0.01,
             f"hidden={cfg['hidden_sizes']}  lr={cfg['lr']}  γ={cfg['gamma']}  "
             f"buffer={cfg['buffer_size']}  batch={cfg['batch_size']}  "
             f"target_update={cfg['target_update_freq']}  ε_decay={cfg['eps_decay_steps']}  "
             f"AUC={train_metrics['train_auc']:.0f}  steps={steps_total}",
             ha='center', fontsize=7.5, color='#555555')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig_path = os.path.join(plots_dir, "dqn_curve.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[Plot] Saved → {fig_path}")

    # ── Save weights ──────────────────────────────────────────────────────────
    pth_path = os.path.join(plots_dir, "dqn_agent.pth")
    torch.save({
        "q_net":         q_net.state_dict(),
        "target_net":    target_net.state_dict(),
        "env_name":      env_name,
        "obs_dim":       obs_dim,
        "n_actions":     n_act,
        "hidden_sizes":  cfg["hidden_sizes"],
        "discrete_obs":  cfg["discrete_obs"],
        "n_states":      cfg["n_states"],
        "train_metrics": train_metrics,
        "eval_metrics":  eval_metrics,
        "eval_mean":     eval_metrics["eval_mean"],
        "success_rate":  eval_metrics["success_rate"],
    }, pth_path)
    print(f"[Model] Saved → {pth_path}")

    return q_net, all_rewards, mean_rewards, train_metrics, eval_metrics


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