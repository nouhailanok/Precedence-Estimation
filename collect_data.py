"""
Collecte de données depuis CartPole-v1 avec Gymnasium
Tâche : Precedence Estimation — f(s_t, a_t) → s_{t+1}

Installation :
    pip install gymnasium torch numpy matplotlib

CartPole-v1 — espace d'état (4D) :
    s[0] : position du chariot       x       ∈ [-4.8, 4.8]
    s[1] : vitesse du chariot        ẋ       ∈ (-∞, +∞)
    s[2] : angle du pendule          θ       ∈ [-0.418 rad, 0.418 rad]
    s[3] : vitesse angulaire         θ̇       ∈ (-∞, +∞)

Actions (discrètes) :
    0 : pousser à gauche
    1 : pousser à droite
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# 1. COLLECTE SIMPLE — politique aléatoire
# ─────────────────────────────────────────────

def collect_data(n_steps: int = 100_000, seed: int = 0) -> tuple:
    """Collecte un nombre FIXE de transitions avec une politique aléatoire.

    Retourne trois tableaux numpy :
        states      (n_steps, 4)  — états s_t
        actions     (n_steps,)    — actions a_t (indices 0/1)
        next_states (n_steps, 4)  — états suivants s_{t+1}

    Note : contrairement à une collecte par épisodes, n_steps garantit exactement
    le même nombre de samples à chaque exécution.
    """
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed)

    states, actions, next_states = [], [], []

    episode = 0
    obs, _ = env.reset(seed=seed)

    for _ in range(n_steps):
        action = int(rng.integers(env.action_space.n))
        next_obs, _, terminated, truncated, _ = env.step(action)

        states.append(obs)
        actions.append(action)
        next_states.append(next_obs)

        if terminated or truncated:
            episode += 1
            obs, _ = env.reset(seed=seed + episode)
        else:
            obs = next_obs

    env.close()

    S = np.asarray(states, dtype=np.float32)
    A = np.asarray(actions, dtype=np.int64)
    SN = np.asarray(next_states, dtype=np.float32)

    print(f"Transitions collectées : {len(S):,} (épisodes traversés: {episode + 1})")
    print(f"  states.shape      : {S.shape}   ← entrée DNN (concat avec action one-hot = 6D)")
    print(f"  actions.shape     : {A.shape}   ← indices 0/1")
    print(f"  next_states.shape : {SN.shape}  ← sortie DNN")
    return S, A, SN


# ─────────────────────────────────────────────
# 2. ANALYSE DU DATASET
# ─────────────────────────────────────────────

def analyze_dataset(S: np.ndarray, A: np.ndarray, SN: np.ndarray):
    """Affiche des statistiques et visualise les distributions."""
    labels = ["x (position)", "ẋ (vitesse)", "θ (angle)", "θ̇ (vit. ang.)"]

    print("\n--- Statistiques des états ---")
    for i, name in enumerate(labels):
        print(f"  {name:20s}  min={S[:,i].min():.3f}  max={S[:,i].max():.3f}  "
              f"mean={S[:,i].mean():.3f}  std={S[:,i].std():.3f}")

    action_counts = np.bincount(A.reshape(-1).astype(np.int64), minlength=2)
    print(f"\nActions : gauche={action_counts[0]:,}  droite={action_counts[1]:,}")

    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    for i, (ax, name) in enumerate(zip(axes, labels)):
        ax.hist(S[:, i], bins=40, alpha=0.7, color="#378ADD", edgecolor="none")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("valeur")
        ax.set_ylabel("fréquence")
    plt.suptitle("Distribution des états — CartPole-v1", fontsize=11)
    plt.tight_layout()
    plt.savefig("cartpole_distributions.png", dpi=120)
    plt.show()


# ─────────────────────────────────────────────
# 3. ONE-HOT + NORMALISATION
# ─────────────────────────────────────────────

def one_hot_actions(A: np.ndarray, n_actions: int = 2) -> np.ndarray:
    """Convertit des actions (N,) ou (N,1) → (N, n_actions)."""
    A = np.asarray(A)
    if A.ndim == 2 and A.shape[1] == 1:
        A = A[:, 0]
    if A.ndim != 1:
        raise ValueError(f"A doit être 1D (N,) ou 2D (N,1). Reçu: {A.shape}")

    A = A.astype(np.int64)
    if A.size and (A.min() < 0 or A.max() >= n_actions):
        raise ValueError(f"Actions hors bornes: min={A.min()}, max={A.max()}, n_actions={n_actions}")

    return np.eye(n_actions, dtype=np.float32)[A]

def normalize(S: np.ndarray):
    """Retourne S_norm, mean, std."""
    mean = S.mean(axis=0, keepdims=True)
    std  = S.std(axis=0, keepdims=True) + 1e-8
    return (S - mean) / std, mean, std


# ─────────────────────────────────────────────
# 4. SPLIT TRAIN / TEST
# ─────────────────────────────────────────────

def split_dataset(S, A, SN, ratio: float = 0.7,ratio_val: float = 0.10 ,ratio_test: float = 0.2, shuffle: bool = True, seed: int | None = None):
    """Retourne (s_train, a_train, sn_train, s_test, a_test, sn_test)."""
    N = len(S)
    if shuffle:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(N)
    else:
        idx = np.arange(N)
    cut = int(N * ratio)
    cut_val = int(N * (ratio + ratio_val))
    tr, val, te = idx[:cut], idx[cut:cut_val], idx[cut_val:]

    print(f"\nSplit : train={len(tr):,}  val={len(val):,}  test={len(te):,}")
    return (S[tr], A[tr], SN[tr],
            S[val], A[val], SN[val],
            S[te], A[te], SN[te])


# ─────────────────────────────────────────────
# 4. VÉRIFICATION RAPIDE
# ─────────────────────────────────────────────

def quick_check():
    """Joue un seul épisode et affiche les 5 premières transitions."""
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=42)
    print("\n--- Vérification : 5 premières transitions ---")
    print(f"{'s_t':45s}  {'a':>2}  {'s_{t+1}'}")
    print("-" * 85)

    for step in range(5):
        action = env.action_space.sample()
        next_obs, _, terminated, truncated, _ = env.step(action)
        s_str  = "[" + ", ".join(f"{v:+.4f}" for v in obs)      + "]"
        sn_str = "[" + ", ".join(f"{v:+.4f}" for v in next_obs) + "]"
        print(f"{s_str:45s}  {action:>2}  {sn_str}")
        if terminated or truncated:
            break
        obs = next_obs
    env.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    quick_check()

    S, A, SN = collect_data(n_steps=100_000, seed=0)
    analyze_dataset(S, A, SN)

    # One-hot actions
    A_oh = one_hot_actions(A)

    # ⚠️  SPLIT AVANT NORMALISATION (pas de data leakage)
    s_tr, a_tr, sn_tr, s_val, a_val, sn_val, s_te, a_te, sn_te = split_dataset(S, A_oh, SN, seed=0)

    # Normalisation : calculer mean/std sur le TRAIN SET UNIQUEMENT
    s_tr_norm, mean, std = normalize(s_tr)
    sn_tr_norm = (sn_tr - mean) / std

    s_val_norm = (s_val - mean) / std
    sn_val_norm = (sn_val - mean) / std

    # Appliquer les MÊMES stats (mean/std du train) au TEST SET
    s_te_norm = (s_te - mean) / std
    sn_te_norm = (sn_te - mean) / std

    # Sauvegarde (optionnel, utile pour réutiliser sans re-collecter)
    np.savez("cartpole_data.npz",
         s_train=s_tr_norm, a_train=a_tr, sn_train=sn_tr_norm,
         s_val=s_val_norm,  a_val=a_val,  sn_val=sn_val_norm,
         s_test=s_te_norm,  a_test=a_te,  sn_test=sn_te_norm,
         mean=mean, std=std)
    # np.savez("normalization_stats.npz", mean=mean, std=std)
    # np.save("s_train.npy",  s_tr_norm)
    # np.save("a_train.npy",  a_tr)
    # np.save("sn_train.npy", sn_tr_norm)
    # np.save("s_val.npy",    s_val_norm)
    # np.save("a_val.npy",    a_val)
    # np.save("sn_val.npy",   sn_val_norm)
    # np.save("s_test.npy",   s_te_norm)
    # np.save("a_test.npy",   a_te)
    # np.save("sn_test.npy",  sn_te_norm)
    # np.save("mean.npy", mean)
    # np.save("std.npy", std)

    print("\nFichiers sauvegardés : s_train.npy, a_train.npy, sn_train.npy, ...")
    print(f"Normalisation : mean={mean}, std={std}")

    print(f"Normalisation : mean={mean.flatten()}, std={std.flatten()}")

    # Rechargement dans un autre script :
    #   S  = np.load("s_train.npy")
    #   A  = np.load("a_train.npy")
    #   SN = np.load("sn_train.npy")
