"""
Collecte de données — LunarLander-v3 (Gymnasium)
Tâche : Precedence Estimation — f(s_t, a_t) → s_{t+1}

Installation :
    pip install gymnasium[box2d] torch numpy matplotlib

LunarLander-v3 — espace d'état (8D) :
    s[0] : position x          ∈ [-1.5, 1.5]
    s[1] : position y          ∈ [-1.5, 1.5]
    s[2] : vitesse x           ∈ [-5, 5]
    s[3] : vitesse y           ∈ [-5, 5]
    s[4] : angle               ∈ [-π, π]
    s[5] : vitesse angulaire   ∈ [-5, 5]
    s[6] : contact pied gauche ∈ {0, 1}  (booléen)
    s[7] : contact pied droit  ∈ {0, 1}  (booléen)

Actions (discrètes, 4 actions) :
    0 : ne rien faire
    1 : moteur gauche
    2 : moteur principal (bas)
    3 : moteur droit

Différences clés vs CartPole :
    - État 8D au lieu de 4D  → entrée DNN = 8 + 4 (one-hot) = 12D
    - 4 actions au lieu de 2 → one-hot de taille 4
    - Reward dense : -100 (crash), +100 (atterrissage), +/-0.3 par step
    - Épisodes plus longs (~200-500 steps) → meilleure couverture des états
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

ENV_ID     = "LunarLander-v3"
OBS_DIM    = 8     # dimension de l'état
N_ACTIONS  = 4     # actions discrètes
INPUT_DIM  = OBS_DIM + N_ACTIONS   # entrée DNN = 12D

DIM_LABELS = [
    "x (pos)",
    "y (pos)",
    "vx (vitesse x)",
    "vy (vitesse y)",
    "angle",
    "vit. angulaire",
    "contact gauche",
    "contact droit",
]


# ─────────────────────────────────────────────────────────────────────────────
# Q-NETWORK  (adapté pour 4 actions)
# ─────────────────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    Q(s, ·) : état (8D) → Q-valeurs pour chaque action (4D).

    Plus grand que CartPole (128 neurones au lieu de 64) car :
    - L'espace d'état est 2× plus grand (8D vs 4D)
    - La dynamique est plus complexe (gravité + moteurs + contact)
    """
    def __init__(self, state_dim: int = OBS_DIM, n_actions: int = N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128,       128), nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# REPLAY BUFFER
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Buffer circulaire (s, a, r, s', done).
    Double rôle : entraîner le DQN + constituer le dataset du prédicteur DNN.
    Capacité plus grande que CartPole car les épisodes sont plus longs.
    """
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((
            np.array(s,  dtype=np.float32),
            int(a),
            float(r),
            np.array(s2, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            np.stack(s).astype(np.float32),
            np.array(a,  dtype=np.int64),
            np.array(r,  dtype=np.float32),
            np.stack(s2).astype(np.float32),
            np.array(d,  dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────────────
# COURBE D'APPRENTISSAGE
# ─────────────────────────────────────────────────────────────────────────────

def plot_learning_curve(
    episode_rewards: list,
    window:    int = 20,
    save_path: str = "lunarlander_learning_curve.png",
):
    """
    Reward par épisode + moyenne glissante.
    Pour LunarLander :
        < -200  → très mauvais (crashes répétés)
        ~0      → survie sans atterrir
        > +200  → atterrissage réussi (objectif)
    """
    if len(episode_rewards) < 2:
        print("Pas assez d'épisodes pour tracer la courbe.")
        return

    rewards  = np.array(episode_rewards)
    episodes = np.arange(1, len(rewards) + 1)

    kernel   = np.ones(window) / window
    smoothed = np.convolve(rewards, kernel, mode="valid")
    smooth_x = np.arange(window, len(rewards) + 1)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(episodes, rewards,   color="#378ADD", alpha=0.25, lw=0.8, label="reward brut")
    ax.plot(smooth_x, smoothed,  color="#1D9E75", lw=2,
            label=f"moyenne glissante ({window} épisodes)")
    ax.axhline(200,  color="#D85A30", linestyle="--", lw=1, alpha=0.8, label="seuil succès (+200)")
    ax.axhline(0,    color="gray",    linestyle=":",  lw=0.8)
    ax.axhline(-100, color="#D85A30", linestyle=":",  lw=0.8, alpha=0.5, label="seuil crash (−100)")

    final_mean = np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards)
    ax.axhline(final_mean, color="#9B59B6", linestyle=":", lw=1,
               label=f"moy. finale = {final_mean:.0f}")

    ax.set_xlabel("Épisode")
    ax.set_ylabel("Reward cumulé")
    ax.set_title("Apprentissage DQN — mixed policy LunarLander-v3")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.show()
    print(f"Sauvegardé → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# COLLECTE MIXED POLICY — DQN ε-greedy
# ─────────────────────────────────────────────────────────────────────────────

def collect_data_dqn(n_steps: int = 200_000, seed: int = 0):
    """
    Collecte mixed policy ε-greedy sur LunarLander-v3.

    Différences vs CartPole :
    ─────────────────────────────────────────────────────────
    | Paramètre       | CartPole     | LunarLander           |
    |─────────────────|──────────────|───────────────────────|
    | n_steps         | 100k         | 200k  (épisodes + longs) |
    | buffer capacity | 50k          | 100k                  |
    | hidden size     | 64           | 128  (état 2× plus grand)|
    | n_actions       | 2            | 4                     |
    | explore_steps   | 80% steps    | 70% steps             |
    | batch_size      | 64           | 128                   |
    ─────────────────────────────────────────────────────────

    Retourne S (n_steps, 8), A (n_steps,), SN (n_steps, 8), episode_rewards
    """
    # ── Reproductibilité ─────────────────────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)

    env = gym.make(ENV_ID)

    # ── DQN ──────────────────────────────────────────────────────────────────
    q_net      = QNetwork()
    target_net = QNetwork()
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=5e-4)   # lr légèrement plus bas
    buffer    = ReplayBuffer(capacity=100_000)

    # ── Hyperparamètres ───────────────────────────────────────────────────────
    gamma         = 0.99
    batch_size    = 128        # plus grand : épisodes plus riches
    epsilon       = 1.0
    epsilon_min   = 0.05
    target_update = 1_000      # plus espacé : épisodes plus longs

    # Calibration du decay : atteindre epsilon_min après 70% des steps
    # (LunarLander apprend plus lentement → garder l'exploration plus longtemps)
    explore_steps = int(0.70 * n_steps)
    epsilon_decay = (epsilon_min / epsilon) ** (1.0 / explore_steps)

    print(f"Environnement     : {ENV_ID}")
    print(f"État              : {OBS_DIM}D   Actions : {N_ACTIONS}")
    print(f"Entrée DNN        : {INPUT_DIM}D  (8 état + 4 one-hot)")
    print(f"epsilon_decay     : {epsilon_decay:.7f}  (ε_min atteint au step ~{explore_steps:,})")
    print(f"Steps à collecter : {n_steps:,}")
    print()

    # ── Collecte ─────────────────────────────────────────────────────────────
    states, actions, next_states = [], [], []
    episode_rewards = []
    current_reward  = 0.0
    episode         = 0

    obs, _ = env.reset(seed=seed)

    for step in range(n_steps):

        # ε-greedy — 4 actions possibles
        if rng.random() < epsilon:
            action = env.action_space.sample()       # 0, 1, 2 ou 3
        else:
            with torch.no_grad():
                q_vals = q_net(torch.tensor(obs, dtype=torch.float32))
                action = int(torch.argmax(q_vals).item())

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done            = terminated or truncated
        current_reward += reward

        # Dataset prédicteur
        states.append(obs)
        actions.append(action)
        next_states.append(next_obs)

        # Buffer DQN
        buffer.push(obs, action, reward, next_obs, done)

        # Reset sans seed fixe après le 1er épisode → diversité
        obs = next_obs if not done else env.reset()[0]

        if done:
            episode_rewards.append(current_reward)
            current_reward = 0.0
            episode       += 1

        # ── Entraînement DQN ─────────────────────────────────────────────────
        if len(buffer) >= batch_size:
            s_b, a_b, r_b, s2_b, d_b = buffer.sample(batch_size)

            s_t  = torch.tensor(s_b,  dtype=torch.float32)
            a_t  = torch.tensor(a_b,  dtype=torch.int64).unsqueeze(1)
            r_t  = torch.tensor(r_b,  dtype=torch.float32)
            s2_t = torch.tensor(s2_b, dtype=torch.float32)
            d_t  = torch.tensor(d_b,  dtype=torch.float32)

            # Q-valeur pour l'action prise
            q_pred = q_net(s_t).gather(1, a_t).squeeze(1)

            # Cible de Bellman avec target network gelé
            with torch.no_grad():
                q_next  = target_net(s2_t).max(1)[0]
                q_target = r_t + gamma * q_next * (1.0 - d_t)

            loss = nn.SmoothL1Loss()(q_pred, q_target)  # Huber plus robuste
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            optimizer.step()

        # ── Sync target network ───────────────────────────────────────────────
        if step > 0 and step % target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

        # ── Décroissance epsilon (par step, pas par épisode) ─────────────────
        # LunarLander : par step est plus stable car les épisodes ont des durées
        # très variables (courts au début, longs quand la politique s'améliore)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # ── Log tous les 20k steps ────────────────────────────────────────────
        if (step + 1) % 20_000 == 0:
            recent = np.mean(episode_rewards[-20:]) if episode_rewards else 0.0
            print(f"Step {step+1:>7,} | ε={epsilon:.3f} | "
                  f"reward moy. (20 ep.) = {recent:+7.1f} | "
                  f"épisodes = {episode}")

    env.close()

    S  = np.asarray(states,      dtype=np.float32)
    A  = np.asarray(actions,     dtype=np.int64)
    SN = np.asarray(next_states, dtype=np.float32)

    print(f"\nTransitions collectées : {len(S):,}")
    print(f"Épisodes              : {episode}")
    print(f"Reward moyen final    : {np.mean(episode_rewards[-50:]):+.1f}")

    evaluate_policy(q_net, n_episodes=20)
    torch.save(q_net.state_dict(), "dqn_lunarlander.pth")
    print("Modèle DQN sauvegardé → dqn_lunarlander.pth")

    plot_learning_curve(episode_rewards)

    return S, A, SN, episode_rewards


# ─────────────────────────────────────────────────────────────────────────────
# ÉVALUATION GREEDY
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_policy(q_net: QNetwork, n_episodes: int = 20, seed: int = 42):
    """
    Évalue la politique greedy (ε=0) sur n_episodes.
    Pour LunarLander : un score > 200 signifie atterrissage réussi.
    """
    env = gym.make(ENV_ID)
    q_net.eval()
    scores = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            with torch.no_grad():
                action = int(q_net(torch.tensor(obs, dtype=torch.float32)).argmax().item())
            obs, r, terminated, truncated, _ = env.step(action)
            done   = terminated or truncated
            total += r
        scores.append(total)

    env.close()
    q_net.train()

    mean_s = np.mean(scores)
    std_s  = np.std(scores)
    status = "✅ atterrissage" if mean_s > 200 else ("⚠️  en progrès" if mean_s > 0 else "❌ crash")
    print(f"Évaluation greedy ({n_episodes} éps) : mean={mean_s:+.1f}  std={std_s:.1f}  {status}")
    return float(mean_s)


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSE DU DATASET
# ─────────────────────────────────────────────────────────────────────────────

def analyze_dataset(S: np.ndarray, A: np.ndarray, SN: np.ndarray):
    """
    Statistiques et distributions du dataset collecté.
    Points d'attention spécifiques à LunarLander :
    - s[6] et s[7] sont binaires {0,1} → distribution bimodale normale
    - s[2] et s[3] (vitesses) peuvent avoir une grande variance en début d'épisode
    - Les deltas (SN - S) devraient être petits sauf pour les contacts (s[6], s[7])
    """
    print("\n--- Statistiques des états (LunarLander-v3) ---")
    print(f"  {'dimension':22s}  {'min':>8}  {'max':>8}  {'mean':>8}  {'std':>8}")
    print("  " + "-" * 60)
    for i, name in enumerate(DIM_LABELS):
        print(f"  {name:22s}  {S[:,i].min():8.3f}  {S[:,i].max():8.3f}"
              f"  {S[:,i].mean():8.3f}  {S[:,i].std():8.3f}")

    counts = np.bincount(A.astype(np.int64), minlength=N_ACTIONS)
    labels_a = ["rien", "gauche", "bas (principal)", "droit"]
    print(f"\nDistribution des actions :")
    for i, (lbl, cnt) in enumerate(zip(labels_a, counts)):
        print(f"  {i} — {lbl:20s} : {cnt:,}  ({100*cnt/len(A):.1f}%)")

    # Delta stats — critique pour choisir entre prédire l'état complet ou le delta
    delta = SN - S
    print(f"\nDelta s_{{t+1}} - s_t par dimension :")
    print(f"  {'dimension':22s}  {'mean':>10}  {'std':>10}")
    for i, name in enumerate(DIM_LABELS):
        print(f"  {name:22s}  {delta[:,i].mean():+10.5f}  {delta[:,i].std():10.5f}")

    # Distributions
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    for i, name in enumerate(DIM_LABELS):
        r, c = i // 4, i % 4
        axes[r, c].hist(S[:, i], bins=50, color="#378ADD", alpha=0.75, edgecolor="none")
        axes[r, c].set_title(name, fontsize=9)
        axes[r, c].set_xlabel("valeur")
        axes[r, c].set_ylabel("fréquence")

    plt.suptitle("Distribution des états — LunarLander-v3", fontsize=11)
    plt.tight_layout()
    plt.savefig("lunarlander_distributions.png", dpi=120)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING — identique à CartPole sauf n_actions=4
# ─────────────────────────────────────────────────────────────────────────────

def one_hot_actions(A: np.ndarray, n_actions: int = N_ACTIONS) -> np.ndarray:
    """
    Indices d'action (N,) → one-hot (N, 4).
    LunarLander a 4 actions au lieu de 2 pour CartPole.
    L'entrée du DNN prédicteur sera donc 8 + 4 = 12D.
    """
    A = np.asarray(A).astype(np.int64)
    if A.min() < 0 or A.max() >= n_actions:
        raise ValueError(f"Actions hors bornes : min={A.min()}, max={A.max()}")
    return np.eye(n_actions, dtype=np.float32)[A]


def split_dataset(
    S, A, SN,
    ratio_train: float = 0.70,
    ratio_val:   float = 0.10,
    ratio_test:  float = 0.20,
    shuffle:     bool  = True,
    seed:        int   = 0,
):
    if not np.isclose(ratio_train + ratio_val + ratio_test, 1.0):
        raise ValueError("Les ratios doivent sommer à 1.0")
    N   = len(S)
    idx = np.random.default_rng(seed).permutation(N) if shuffle else np.arange(N)
    c1  = int(N * ratio_train)
    c2  = int(N * (ratio_train + ratio_val))
    tr, val, te = idx[:c1], idx[c1:c2], idx[c2:]
    print(f"Split : train={len(tr):,}  val={len(val):,}  test={len(te):,}")
    return S[tr], A[tr], SN[tr], S[val], A[val], SN[val], S[te], A[te], SN[te]


def normalize_states(s_tr, sn_tr, s_val, sn_val, s_te, sn_te):
    """
    Fit mean/std sur le train uniquement.

    Note LunarLander : s[6] et s[7] sont binaires {0,1}.
    Leur std sera ~0.5, donc la normalisation les ramènera à [-1, 1].
    Ce n'est pas un problème — le DNN peut apprendre cette structure.
    """
    mean = s_tr.mean(axis=0, keepdims=True)
    std  = s_tr.std(axis=0,  keepdims=True)
    std  = np.where(std < 1e-8, 1.0, std)   # évite division par zéro sur dims constantes
    norm = lambda X: (X - mean) / std
    return (norm(s_tr), norm(sn_tr),
            norm(s_val), norm(sn_val),
            norm(s_te),  norm(sn_te),
            mean, std)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("Collecte LunarLander-v3 — mixed policy DQN")
    print(f"Entrée DNN prédicteur : {INPUT_DIM}D  (8 état + 4 one-hot)")
    print("=" * 60)

    # ── 1. Collecte ───────────────────────────────────────────────────────────
    S, A, SN, ep_rewards = collect_data_dqn(n_steps=200_000, seed=0)

    # ── 2. Analyse ────────────────────────────────────────────────────────────
    analyze_dataset(S, A, SN)

    # ── 3. One-hot (4 actions, pas 2) ─────────────────────────────────────────
    A_oh = one_hot_actions(A, n_actions=N_ACTIONS)
    print(f"\nOne-hot shape : {A_oh.shape}  (exemple : {A_oh[0]})")

    # ── 4. Split AVANT normalisation ─────────────────────────────────────────
    (s_tr, a_tr, sn_tr,
     s_val, a_val, sn_val,
     s_te, a_te, sn_te) = split_dataset(S, A_oh, SN, seed=0)

    # ── 5. Normaliser sur le train uniquement ─────────────────────────────────
    (s_tr_n, sn_tr_n,
     s_val_n, sn_val_n,
     s_te_n, sn_te_n,
     mean, std) = normalize_states(s_tr, sn_tr, s_val, sn_val, s_te, sn_te)

    # ── 6. Sauvegarder ────────────────────────────────────────────────────────
    np.savez("lunarlander_data_mixed_policy.npz",
             s_train=s_tr_n,  a_train=a_tr,   sn_train=sn_tr_n,
             s_val=s_val_n,   a_val=a_val,     sn_val=sn_val_n,
             s_test=s_te_n,   a_test=a_te,     sn_test=sn_te_n,
             mean=mean,       std=std)

    print("\n" + "=" * 60)
    print("Fichiers sauvegardés :")
    print("  lunarlander_data_mixed_policy.npz")
    print("  dqn_lunarlander.pth")
    print("  lunarlander_learning_curve.png")
    print("  lunarlander_distributions.png")
    print()
    print("Formes finales :")
    print(f"  s_train  : {s_tr_n.shape}   état normalisé")
    print(f"  a_train  : {a_tr.shape}   one-hot 4 actions")
    print(f"  sn_train : {sn_tr_n.shape}   état suivant normalisé")
    print()
    print(f"  Entrée DNN prédicteur : {s_tr_n.shape[1] + a_tr.shape[1]}D")
    print(f"  Sortie DNN prédicteur : {sn_tr_n.shape[1]}D")
    print("=" * 60)