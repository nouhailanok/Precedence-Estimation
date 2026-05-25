"""
dyna_q_ablation.py — Expérience ablation pour l'effet du world model (Dyna-Q vs DQN)
Compare :
  - DQN classique (N_DYNA=0, pur model-free)
  - Dyna-Q (N_DYNA>0, world model plugged in)
Pour chaque, learning curve, score greedy, courbe superposée pour publication.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import gymnasium as gym
import os
import random
from collections import deque

# ==== CONFIG EXPÉRIMENTALE ====
ENV_NAME         = "CartPole-v1"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORLD_MODEL_PATH = os.path.join(BASE_DIR, "checkpoints_compare/Tiny_seed0.pth")
WORLD_MODEL_STATS= "cartpole_data_mixed_policy.npz"
# === CONFIG
PLOTS_ABLATION_DIR = os.path.join(BASE_DIR, "ABLATIONDyna_Q_plots")
os.makedirs(PLOTS_ABLATION_DIR, exist_ok=True)

N_DYNA_LIST      = [0,5, 10]    # DQN pur / Dyna-Q  (ajoute p.ex [0,1,5,10])
N_SEEDS          = 3      # Runs par config pour moyenne/robustesse
N_EPISODES       = 300
MAX_STEPS_PER_EP = 500
N_ACTIONS        = 2
BUFFER_CAPACITY  = 40000
BATCH_SIZE       = 64
GAMMA            = 0.99
LR               = 3e-4
TARGET_FREQ      = 300
EPS_START        = 1.0
EPS_END          = 0.02
EPS_DECAY        = 0.995
SEED_BASE        = 42
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== World model (doit être strictement la même archi que pour train) ====
class TransitionDNN(nn.Module):
    def __init__(self, state_dim=4, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, state_dim),
        )
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

def load_world_model(path, mean_std_path=WORLD_MODEL_STATS):
    wm = TransitionDNN().to(DEVICE)
    wm.load_state_dict(torch.load(path, map_location=DEVICE))
    wm.eval()
    for p in wm.parameters(): p.requires_grad = False
    stats = np.load(mean_std_path)
    mean = stats["mean"].astype(np.float32).reshape(-1)
    std = stats["std"].astype(np.float32).reshape(-1)
    return wm, mean, std

def normalize_state(s, mean, std):
    s = np.asarray(s, dtype=np.float32).reshape(-1)
    return (s - mean) / std
def denormalize(s_norm, mean, std):
    return s_norm * std + mean
def one_hot(a, n=N_ACTIONS):
    arr = np.zeros(n, dtype=np.float32)
    arr[a] = 1.0
    return arr
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

# ==== Q Network, Buffer
class QNet(nn.Module):
    def __init__(self, in_dim, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_out)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=BUFFER_CAPACITY):
        self.buffer = deque(maxlen=capacity)
    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            np.stack(s), np.array(a), np.array(r, dtype=np.float32),
            np.stack(s2), np.array(d, dtype=np.float32)
        )
    def __len__(self):
        return len(self.buffer)

def safe_torch(x, dtype, device=DEVICE):
    return torch.tensor(np.asarray(x), dtype=dtype, device=device)

# ==== Entraînement UNE courbe
def run_dynaq(N_DYNA, seed):
    set_seed(seed)
    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    n_act   = env.action_space.n
    # world model chargé SEULEMENT si NDYNA>0
    if N_DYNA > 0:
        wm, mean, std = load_world_model(WORLD_MODEL_PATH)
    else:
        wm = None
        mean = std = None

    qnet = QNet(obs_dim, n_act).to(DEVICE)
    q_target = QNet(obs_dim, n_act).to(DEVICE)
    q_target.load_state_dict(qnet.state_dict())
    optimizer = optim.Adam(qnet.parameters(), lr=LR)
    buffer = ReplayBuffer()
    criterion = nn.MSELoss()

    rewards, rewards_mean25 = [], []
    eps = EPS_START
    steps_total = 0

    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=SEED_BASE+ep+seed*100)
        done = False
        ep_reward = 0

        for t in range(MAX_STEPS_PER_EP):
            # ε-greedy
            if np.random.rand() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    qvals = qnet(safe_torch(obs, torch.float32).unsqueeze(0)).cpu()
                    action = int(torch.argmax(qvals).item())
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(obs, action, reward, next_obs, done)
            ep_reward += reward

            # learning
            if len(buffer) >= BATCH_SIZE:
                s, a, r, s2, d = buffer.sample(BATCH_SIZE)
                s    = safe_torch(s, torch.float32)
                a_t  = safe_torch(a, torch.int64).unsqueeze(1)
                r    = safe_torch(r, torch.float32)
                s2   = safe_torch(s2, torch.float32)
                d    = safe_torch(d, torch.float32)
                q_eval = qnet(s).gather(1, a_t).squeeze()
                with torch.no_grad():
                    q_next = q_target(s2).max(1)[0]
                    target = r + GAMMA * q_next * (1 - d)
                loss = criterion(q_eval, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # --- Dyna-Q
            if N_DYNA > 0 and wm is not None and len(buffer) > BATCH_SIZE:
                for _ in range(N_DYNA):
                    idx = np.random.randint(max(0, len(buffer)-1500), len(buffer))
                    s_seed, _, _, _, _ = buffer.buffer[idx]
                    s_norm = normalize_state(s_seed, mean, std)
                    if np.random.rand() < 0.6:
                        a_dyna = qnet(safe_torch(s_seed, torch.float32).unsqueeze(0)).argmax().item()
                    else:
                        a_dyna = np.random.randint(0, n_act)
                    a_oh    = one_hot(a_dyna)
                    s_in = safe_torch(np.atleast_2d(s_norm), torch.float32)
                    a_in = safe_torch(np.atleast_2d(a_oh),   torch.float32)
                    with torch.no_grad():
                        delta_pred = wm(s_in, a_in).cpu().numpy()[0]
                        s2_norm = s_norm + delta_pred
                        s2_fake = denormalize(s2_norm, mean, std)
                        s2_fake = np.array(s2_fake).reshape(-1)
                    x, theta = s2_fake[0], s2_fake[2]
                    done_fake = x < -2.4 or x > 2.4 or theta < -0.209 or theta > 0.209
                    r_fake = 0.0 if done_fake else 1.0
                    buffer.push(
                        s_seed, a_dyna, r_fake, s2_fake, done_fake
                    )

            obs = next_obs
            if done: break
            steps_total += 1

            if steps_total > 0 and steps_total % TARGET_FREQ == 0:
                q_target.load_state_dict(qnet.state_dict())
        rewards.append(ep_reward)
        rewards_mean25.append(np.mean(rewards[-25:]))
        eps = max(EPS_END, eps * EPS_DECAY)
    env.close()
    # Score final greedy policy :
    eval_env = gym.make(ENV_NAME)
    obs, _ = eval_env.reset(seed=SEED_BASE+1000+seed*222)
    total_reward, done = 0, False
    qnet.eval()
    while not done:
        with torch.no_grad():
            qvals = qnet(safe_torch(obs, torch.float32).unsqueeze(0))
            action = int(torch.argmax(qvals).item())
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        total_reward += reward
    eval_env.close()
    return np.array(rewards), np.array(rewards_mean25), total_reward

# ==== EXPÉRIENCE MULTI-RUNS
if __name__ == "__main__":
    results = {}
    for ndyna in N_DYNA_LIST:
        print("------  N_DYNA =", ndyna, "------")
        all_curves = []
        all_curves25 = []
        test_rewards = []
        for seed in range(N_SEEDS):
            rew, rew25, test = run_dynaq(ndyna, SEED_BASE+seed)
            all_curves.append(rew)
            all_curves25.append(rew25)
            test_rewards.append(test)
            print(f"    Seed {seed}: Final greedy test reward = {test}")
        mean_curve = np.mean(all_curves, axis=0)
        std_curve = np.std(all_curves, axis=0)
        mean25 = np.mean(all_curves25, axis=0)
        std25 = np.std(all_curves25, axis=0)
        final_test_mean, final_test_std = np.mean(test_rewards), np.std(test_rewards)
        results[ndyna] = {
            "mean_curve": mean_curve,
            "std_curve": std_curve,
            "mean25": mean25,
            "std25": std25,
            "test_rewards": test_rewards,
            "test_mean": final_test_mean,
            "test_std": final_test_std,
        }
        print(f"Résultat NDYNA={ndyna}: final mean25={mean25[-1]:.1f} | greedy test={final_test_mean:.1f}±{final_test_std:.1f}")

    # ==== Plotting
    colors = ["#378ADD", "#D85A30", "#2ECC71", "#9B59B6", "#9B59B6"]
    plt.figure(figsize=(11,6))
    for i, ndyna in enumerate(N_DYNA_LIST):
        m, s = results[ndyna]["mean25"], results[ndyna]["std25"]
        label = f"DQN" if ndyna==0 else f"Dyna-Q (N_DYNA={ndyna})"
        plt.plot(m, label=label, color=colors[i%len(colors)], linewidth=2)
        plt.fill_between(range(len(m)), m-s, m+s, color=colors[i%len(colors)], alpha=0.18)
    plt.xlabel("Episode")
    plt.ylabel("Mean reward over 25 episodes")
    plt.title("Dyna-Q vs DQN — Learning Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_ABLATION_DIR,"abl_dynaq_dqn_learning_curve.png"), dpi=130)
    plt.show()

    # Résumé ablation
    print("\nRésumé ablation Dyna-Q/DQN :")
    for ndyna in N_DYNA_LIST:
        tag = "DQN" if ndyna==0 else f"Dyna-Q ({ndyna})"
        print(f"{tag:<12s}: Final greedy test mean25={results[ndyna]['mean25'][-1]:.2f} | Greedy test reward = {results[ndyna]['test_mean']:.1f}±{results[ndyna]['test_std']:.1f}")
    print("Courbe sauvegardée → abl_dynaq_dqn_learning_curve.png")