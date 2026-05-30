"""
dyna_q_agent.py — Dyna-Q CartPole-v1 (best practices)
- World model plug-in (Δs), input safe, buffer balancé, Dyna transitions intelligentes
- Compatible avec .pth et mean/std sauvés (remplis par train_dnn_mixed_policy)
- Contrôle du ratio réel/fake, monitoring, seed, robustesse
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import gymnasium as gym
import random
from collections import deque
import os

# === CONFIG
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORLD_MODEL_PATH = os.path.join(BASE_DIR, "checkpoints_compare/Tiny_seed0.pth")
ENV_NAME         = "CartPole-v1"
PLOTS_CREATE_DIR = os.path.join(BASE_DIR, "CREATEAGENT_plots")
N_DYNA           = 0           # Dyna steps par pas réel
N_EPISODES       = 450
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
SEED             = 42
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(PLOTS_CREATE_DIR, exist_ok=True)

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# ==== World model Δs ====
class WorldModel(nn.Module):
    def __init__(self, state_dim=4, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, state_dim)
        )
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
    def predict_next(self, state, action):
        return state + self.forward(state, action)

# === World Model Utilities
# def load_world_model(path, mean_std_path="cartpole_data_mixed_policy.npz"):
#     # load world model (Δs)
#     wm = WorldModel().to(DEVICE)
#     wm.load_state_dict(torch.load(path, map_location="cpu"))
#     wm.eval()
#     for p in wm.parameters(): p.requires_grad = False
#     # Load normalization stats (mean,std used at world model training time)
#     stats = np.load(mean_std_path)
#     mean = stats["mean"].astype(np.float32).reshape(-1)
#     std = stats["std"].astype(np.float32).reshape(-1)
#     return wm, mean, std

def load_world_model(path, mean_std_path="cartpole_data_mixed_policy.npz"):
    # load world model (Δs)
    wm = WorldModel().to(DEVICE)
    wm.load_state_dict(torch.load(path, map_location=DEVICE))
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False
    # Load normalization stats (mean,std used at world model training time)

    if mean_std_path and os.path.exists(mean_std_path):
        stats = np.load(mean_std_path)
        mean = stats["mean"].astype(np.float32)
        std  = stats["std"].astype(np.float32)
    else:
        # CartPole : normalisation identité si stats absentes
        mean = np.zeros(4, dtype=np.float32)
        std  = np.ones(4,  dtype=np.float32)
        print("[WM] Attention : mean/std non trouvés, normalisation désactivée.")
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

# === DQN + REPLAY BUFFER
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

# class ReplayBuffer:
#     def __init__(self, capacity=BUFFER_CAPACITY):
#         self.buffer = deque(maxlen=capacity)
#         self.real_count = 0    # optionnel : suivi réel/fake
#         self.synth_count = 0
#     def push(self, s, a, r, s2, done, is_synthetic=False):
#         self.buffer.append((s, a, r, s2, done, is_synthetic))
#         if is_synthetic: self.synth_count += 1
#         else: self.real_count += 1
#     def sample(self, batch_size, synthetic_ratio=None):
#         """
#         Option : impose un ratio de fake/real dans le batch (avancé)
#         """
#         batch = random.sample(self.buffer, batch_size)
#         s, a, r, s2, d, syn = zip(*batch)
#         return (
#             np.stack(s), np.array(a), np.array(r, dtype=np.float32),
#             np.stack(s2), np.array(d, dtype=np.float32)
#         )
#     def __len__(self):
#         return len(self.buffer)
#     def real_fraction(self):
#         n = self.real_count + self.synth_count
#         return self.real_count / (n+1e-6)


class ReplayBuffer:
    def __init__(self, capacity=BUFFER_CAPACITY):
        self.buffer = deque(maxlen=capacity)
        self._list  = []          # miroir pour indexation rapide
        self.real_count = 0
        self.synth_count = 0

    def push(self, s, a, r, s2, done, is_synthetic=False):
        if len(self.buffer) == self.buffer.maxlen:
            self._list.pop(0)     # sync avec le deque
        t = (s, a, r, s2, done, is_synthetic)
        self.buffer.append(t)
        self._list.append(t)
        if is_synthetic: self.synth_count += 1
        else:            self.real_count  += 1

    def sample_seed_state(self, recent_k=2000):
        """Tire un état seed dans les recent_k dernières transitions."""
        lo  = max(0, len(self._list) - recent_k)
        idx = random.randint(lo, len(self._list) - 1)
        return self._list[idx][0]   # state uniquement

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d, _ = zip(*batch)
        return (np.stack(s), np.array(a), np.array(r, dtype=np.float32),
                np.stack(s2), np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)
    
    def real_fraction(self):
        n = self.real_count + self.synth_count
        return self.real_count / (n+1e-6)
    

def safe_torch(x, dtype, device=DEVICE):
    return torch.tensor(np.asarray(x), dtype=dtype, device=device)

# === MAIN AGENT
def train_dyna():
    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    n_act   = env.action_space.n

    wm, mean, std = load_world_model(WORLD_MODEL_PATH)
    print("World Model chargé : Δs (frozen), mean/std:", mean, std)

    qnet = QNet(obs_dim, n_act).to(DEVICE)
    q_target = QNet(obs_dim, n_act).to(DEVICE)
    q_target.load_state_dict(qnet.state_dict())
    optimizer = optim.Adam(qnet.parameters(), lr=LR)
    buffer = ReplayBuffer()
    criterion = nn.MSELoss()

    all_rewards, mean_rewards = [], []
    eps = EPS_START
    steps_total = 0

    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=SEED+ep)
        done = False
        ep_reward = 0

        for t in range(MAX_STEPS_PER_EP):
            # Épsi-greedy avec QNet
            if np.random.rand() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    qvals = qnet(safe_torch(obs, torch.float32).unsqueeze(0)).cpu()
                    action = int(torch.argmax(qvals).item())

            # -- Réel env
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(obs, action, reward, next_obs, done, is_synthetic=False)
            ep_reward += reward

            # -- Learn (contenu mixé)
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

            # -- Dyna: pour chaque pas réel, on génère N_DYNA fake transitions
            if N_DYNA > 0 and len(buffer) > BATCH_SIZE:
                for _ in range(N_DYNA):
                    # → État seedé : récent ou au hasard mais pas trop vieux
                    s_seed = buffer.sample_seed_state()
                    s_norm = normalize_state(s_seed, mean, std)
                    # → Action : soft-greedy sur QNet mais injecte un peu de random (mix)
                    # if np.random.rand() < eps:
                    #     a_dyna = qnet(safe_torch(s_seed, torch.float32).unsqueeze(0)).argmax().item()
                    # else:
                    #     a_dyna = np.random.randint(0, n_act)

                    if np.random.rand() < eps:
                        a_dyna = np.random.randint(0, n_act)
                    else:
                        a_dyna = qnet(safe_torch(s_seed, torch.float32).unsqueeze(0)).argmax().item()

                    a_oh    = one_hot(a_dyna)
                    # → WM : toujours batch shape [1,6]
                    s_in = safe_torch(np.atleast_2d(s_norm), torch.float32)
                    a_in = safe_torch(np.atleast_2d(a_oh),   torch.float32)
                    # Predict delta + assemble
                    with torch.no_grad():
                        delta_pred = wm(s_in, a_in).cpu().numpy()[0]
                        s2_norm = s_norm + delta_pred
                        s2_fake = denormalize(s2_norm, mean, std)
                        s2_fake = np.array(s2_fake).reshape(-1)
                    # - Done : CartPole-v1 limits sur l'état (comme dans collect scripts)
                    x, theta = s2_fake[0], s2_fake[2]
                    done_fake = x < -2.4 or x > 2.4 or theta < -0.209 or theta > 0.209
                    r_fake = 0.0 if done_fake else 1.0
                    buffer.push(
                        s_seed, a_dyna, r_fake, s2_fake, done_fake,
                        is_synthetic=True
                    )

            obs = next_obs

            steps_total += 1
            if steps_total % TARGET_FREQ == 0:
                q_target.load_state_dict(qnet.state_dict())
            if done:
                break

        all_rewards.append(ep_reward)
        mean_rewards.append(np.mean(all_rewards[-25:]))
        eps = max(EPS_END, eps * EPS_DECAY)

        if (ep+1) % 10 == 0:
            ratio = buffer.real_fraction()
            print(f"Ep {ep+1:03d}  R={ep_reward:.1f}  MeanR25={mean_rewards[-1]:.2f}  eps={eps:.3f} "
                  f"| Buffer real/fake: {100*ratio:.1f}% réal.")

    env.close()
    # Plot
    plt.plot(all_rewards, label="Return")
    plt.plot(mean_rewards, label="Mean25")
    plt.grid(alpha=0.3)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Dyna-Q CartPole (N_DYNA={N_DYNA})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_CREATE_DIR, f"dynaq_curve_cl_correction_{N_DYNA}.png"))
    plt.show()
    torch.save(qnet.state_dict(), os.path.join(PLOTS_CREATE_DIR, f"dynaq_agent_cl_correction_{N_DYNA}.pth"))
    print("Script Dyna-Q terminé.")

    # Evaluate greedy
    eval_env = gym.make(ENV_NAME)
    obs, _ = eval_env.reset(seed=SEED+111)
    total_reward = 0
    done = False
    qnet.eval()
    while not done:
        with torch.no_grad():
            qvals = qnet(safe_torch(obs, torch.float32).unsqueeze(0))
            action = int(torch.argmax(qvals).item())
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        total_reward += reward
    print(f"Reward final agent greedy = {total_reward}")
    eval_env.close()

if __name__ == "__main__":
    train_dyna()