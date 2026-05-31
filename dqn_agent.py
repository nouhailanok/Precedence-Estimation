"""
dqn_agent.py — Pure DQN baseline for CartPole-v1
=================================================

Design decisions (mirroring the PPO file's spirit):
  - Separate online Q-network and frozen target Q-network (hard update every C steps)
  - Experience replay buffer (uniform sampling)
  - ε-greedy exploration with linear decay
  - Separate Adam optimiser for the Q-network
  - Per-episode tracking for a clean learning curve

Output
------
    DQN_plots/
        dqn_curve.png
        dqn_agent.pth
"""

import os
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
# PATHS
# ─────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "DQN_plots")
ENV_NAME  = "CartPole-v1"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
N_EPISODES        = 600
MAX_STEPS         = 500

HIDDEN_SIZES      = (64, 64)

BUFFER_SIZE       = 50_000
BATCH_SIZE        = 64

LR                = 1e-3
GAMMA             = 0.99

TARGET_UPDATE_FREQ = 200

EPS_START         = 1.0
EPS_END           = 0.01
EPS_DECAY_STEPS   = 10_000

WARMUP_STEPS      = 1_000

SEED   = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ─────────────────────────────────────────────
# Q-NETWORK
# ─────────────────────────────────────────────
def build_mlp(in_dim: int, hidden_sizes: tuple, out_dim: int) -> nn.Sequential:
    layers, prev = [], in_dim
    for h in hidden_sizes:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int,
                 hidden: tuple = HIDDEN_SIZES):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ─────────────────────────────────────────────
# REPLAY BUFFER
# ─────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int = BUFFER_SIZE):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buf.append((
            np.asarray(obs,      dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_obs, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int = BATCH_SIZE):
        batch = random.sample(self.buf, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            torch.tensor(np.stack(obs),      dtype=torch.float32, device=DEVICE),
            torch.tensor(actions,            dtype=torch.int64,   device=DEVICE),
            torch.tensor(rewards,            dtype=torch.float32, device=DEVICE),
            torch.tensor(np.stack(next_obs), dtype=torch.float32, device=DEVICE),
            torch.tensor(dones,              dtype=torch.float32, device=DEVICE),
        )

    def __len__(self):
        return len(self.buf)


# ─────────────────────────────────────────────
# ε-GREEDY POLICY
# ─────────────────────────────────────────────
def get_epsilon(step: int) -> float:
    frac = min(step / EPS_DECAY_STEPS, 1.0)
    return EPS_START + frac * (EPS_END - EPS_START)


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
               replay: ReplayBuffer) -> float:
    obs_b, act_b, rew_b, next_obs_b, done_b = replay.sample()

    q_values = q_net(obs_b).gather(1, act_b.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q    = target_net(next_obs_b).max(dim=1).values
        td_target = rew_b + GAMMA * next_q * (1.0 - done_b)

    loss = F.mse_loss(q_values, td_target)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
    optimizer.step()

    return loss.item()


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
def train_dqn():
    env     = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    n_act   = env.action_space.n

    q_net      = QNetwork(obs_dim, n_act, HIDDEN_SIZES).to(DEVICE)
    target_net = QNetwork(obs_dim, n_act, HIDDEN_SIZES).to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    replay    = ReplayBuffer(BUFFER_SIZE)

    print(f"[DQN] Pure baseline — no world model")
    print(f"      N_EPISODES={N_EPISODES} | BUFFER={BUFFER_SIZE} | BATCH={BATCH_SIZE}")
    print(f"      LR={LR} | GAMMA={GAMMA} | TARGET_UPDATE={TARGET_UPDATE_FREQ}")
    print(f"      ε: {EPS_START} → {EPS_END} over {EPS_DECAY_STEPS} steps")

    all_rewards, cumulative_mean = [], []
    losses      = []
    steps_total = 0

    for episode in range(1, N_EPISODES + 1):
        obs, _ = env.reset(seed=SEED + episode)
        ep_return = 0.0

        for _ in range(MAX_STEPS):
            epsilon = get_epsilon(steps_total)
            action  = select_action(obs, q_net, epsilon, n_act)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay.push(obs, action, reward, next_obs, done)
            obs          = next_obs
            ep_return   += reward
            steps_total += 1

            if len(replay) >= WARMUP_STEPS:
                loss = dqn_update(q_net, target_net, optimizer, replay)
                losses.append(loss)

            if steps_total % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(q_net.state_dict())

            if done:
                break

        all_rewards.append(ep_return)
        cumulative_mean.append(np.mean(all_rewards))

        if episode % 50 == 0:
            print(f"Episode {episode:03d}/{N_EPISODES}  "
                  f"Return={ep_return:.1f}  "
                  f"MeanAll={cumulative_mean[-1]:.2f}  "
                  f"ε={get_epsilon(steps_total):.3f}  "
                  f"Steps={steps_total}")

    env.close()

    # ── Single reward plot ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(all_rewards,      alpha=0.35, label="Episode Return")
    ax.plot(cumulative_mean, linewidth=2,  label="Mean (All Episodes)")

    # Annotate the final mean reward as a value on the plot
    final_mean = cumulative_mean[-1]
    ax.axhline(y=final_mean, color='steelblue', linestyle='--', alpha=0.6, linewidth=1)
    ax.text(len(cumulative_mean) - 1, final_mean + 5,
            f"Final Mean (All): {final_mean:.1f}",
            ha='right', va='bottom', color='steelblue', fontsize=10)

    ax.grid(alpha=0.3)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("DQN — CartPole-v1 (no Precedence estimation)")
    ax.legend()

    plt.tight_layout()
    fig_path = os.path.join(PLOTS_DIR, "dqn_curve.png")
    plt.savefig(fig_path, dpi=150)
    plt.show()
    print(f"[Plot] Saved → {fig_path}")

    # ── Save weights ─────────────────────────────────────────────────────────
    pth_path = os.path.join(PLOTS_DIR, "dqn_agent.pth")
    torch.save({
        "q_net":      q_net.state_dict(),
        "target_net": target_net.state_dict(),
    }, pth_path)
    print(f"[Model] Saved → {pth_path}")

    # ── Greedy evaluation ────────────────────────────────
    eval_rewards = []
    eval_env     = gym.make(ENV_NAME)
    q_net.eval()

    for i in range(10):   # 10 greedy episodes → stable mean
        obs_e, _ = eval_env.reset(seed=SEED + 200 + i)
        ep_r, done_e = 0.0, False
        while not done_e:
            with torch.no_grad():
                obs_t = torch.tensor(obs_e, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                a_e   = int(q_net(obs_t).argmax(dim=1).item())
            obs_e, r_e, term_e, trunc_e, _ = eval_env.step(a_e)
            done_e = term_e or trunc_e
            ep_r  += r_e
        eval_rewards.append(ep_r)

    eval_env.close()
    print(f"[Eval] Greedy mean reward (10 episodes) = {np.mean(eval_rewards):.2f}  "
          f"(min={min(eval_rewards):.0f}, max={max(eval_rewards):.0f})")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    train_dqn()