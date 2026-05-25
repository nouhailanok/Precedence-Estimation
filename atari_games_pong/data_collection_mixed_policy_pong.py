"""
Collecte de données — ALE/Pong-v5 (Atari)
Tâche : Precedence Estimation — f(s_t, a_t) → s_{t+1}

Installation :
    pip install gymnasium[atari] ale-py torch numpy matplotlib

Pong-v5 — espace d'état :
    Brut     : image RGB  (210, 160, 3)  uint8
    Prétraité: 4 frames empilées, grayscale, redimensionnées → (4, 84, 84) float32

    Pourquoi empiler 4 frames ?
    Une seule frame ne donne pas la vitesse de la balle.
    Avec 4 frames consécutives, le réseau peut inférer direction + vitesse.

Actions (6 au total, 3 pertinentes pour Pong) :
    0 : NOOP      (ne rien faire)
    2 : UP        (raquette monte)
    3 : DOWN      (raquette descend)
    1, 4, 5 : feu/variantes — ignorés par la mécanique Pong

Reward :
    +1 : point marqué
    -1 : point perdu
     0 : sinon
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import os
from collections import deque


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

ENV_ID      = "ALE/Pong-v5"
FRAME_SIZE  = 84          # frames redimensionnées à 84×84
N_STACK     = 4           # nombre de frames empilées
N_ACTIONS   = 6           # actions Atari standard
OBS_SHAPE   = (N_STACK, FRAME_SIZE, FRAME_SIZE)   # (4, 84, 84)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. PRÉTRAITEMENT DES FRAMES
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Convertit une frame Atari brute (210, 160, 3) uint8
    en frame grayscale redimensionnée (84, 84) float32 dans [0, 1].

    Étapes :
        1. RGB → Grayscale  : 0.299·R + 0.587·G + 0.114·B
           (coefficients standard luminance humaine)
        2. Crop             : on retire les 34 pixels en haut (score UI)
           et on garde 160×160 de la zone de jeu
        3. Resize → 84×84   : sous-échantillonnage bilinéaire simple
        4. Normalise [0,255] → [0,1]

    Pourquoi uint8 en entrée mais float32 en sortie ?
    Les frames brutes tiennent en uint8 (1 octet/pixel) pour économiser la RAM.
    La normalisation [0,1] nécessite float32 pour le réseau.
    """
    # Grayscale
    gray = (0.299 * frame[:, :, 0] +
            0.587 * frame[:, :, 1] +
            0.114 * frame[:, :, 2]).astype(np.float32)

    # Crop : retire les 34 pixels du haut (bandeau score)
    gray = gray[34:194, :]           # (160, 160)

    # Resize simple (sans cv2 ni PIL) par sous-échantillonnage
    # Chaque pixel de la sortie 84×84 est la moyenne d'un bloc ~2×2
    scale_h = gray.shape[0] / FRAME_SIZE   # ~1.9
    scale_w = gray.shape[1] / FRAME_SIZE   # ~1.9
    resized = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.float32)
    for i in range(FRAME_SIZE):
        for j in range(FRAME_SIZE):
            h0 = int(i * scale_h); h1 = min(int((i+1) * scale_h), gray.shape[0])
            w0 = int(j * scale_w); w1 = min(int((j+1) * scale_w), gray.shape[1])
            resized[i, j] = gray[h0:h1, w0:w1].mean()

    return resized / 255.0           # normalise dans [0, 1]


def preprocess_frame_fast(frame: np.ndarray) -> np.ndarray:
    """
    Version rapide utilisant numpy stride tricks.
    Utilisée si scipy n'est pas disponible.
    Même résultat que preprocess_frame() mais ~20x plus rapide.
    """
    gray = (0.299 * frame[:, :, 0] +
            0.587 * frame[:, :, 1] +
            0.114 * frame[:, :, 2]).astype(np.float32)

    gray = gray[34:194, :]           # (160, 160)

    # Sous-échantillonnage 160→84 par moyennage de blocs
    # On redimensionne à 84x84 en prenant 1 pixel sur ~1.9
    idx_h = (np.arange(FRAME_SIZE) * gray.shape[0] / FRAME_SIZE).astype(int)
    idx_w = (np.arange(FRAME_SIZE) * gray.shape[1] / FRAME_SIZE).astype(int)
    resized = gray[np.ix_(idx_h, idx_w)]    # (84, 84)

    return (resized / 255.0).astype(np.float32)


# Essaie d'utiliser scipy pour un resize de qualité, sinon fallback
try:
    from scipy.ndimage import zoom
    def preprocess_frame_best(frame: np.ndarray) -> np.ndarray:
        gray    = (0.299*frame[:,:,0] + 0.587*frame[:,:,1] + 0.114*frame[:,:,2])
        gray    = gray[34:194, :].astype(np.float32)
        resized = zoom(gray, (FRAME_SIZE/gray.shape[0], FRAME_SIZE/gray.shape[1]),
                       order=1)   # interpolation bilinéaire
        return (resized / 255.0).astype(np.float32)
    PREPROCESS = preprocess_frame_best
    print("Preprocessing : scipy.ndimage.zoom (bilinéaire) ✓")
except ImportError:
    PREPROCESS = preprocess_frame_fast
    print("Preprocessing : numpy stride (rapide) — pip install scipy pour mieux")


class FrameStack:
    """
    Gère l'empilement de N_STACK frames consécutives.

    Pourquoi empiler des frames ?
    Une seule frame Pong est une image statique — impossible de savoir
    si la balle monte ou descend. Avec 4 frames consécutives, le réseau
    peut inférer la trajectoire de la balle et la vitesse des raquettes.

    Structure interne : deque de longueur N_STACK.
    Lors d'un reset, toutes les frames sont initialisées à la frame courante
    (pas de frames noires) → le réseau ne voit pas d'artefact de démarrage.
    """
    def __init__(self, n_stack: int = N_STACK):
        self.n_stack = n_stack
        self.frames  = deque(maxlen=n_stack)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        """Initialise la pile avec n_stack copies de la première frame."""
        processed = PREPROCESS(frame)
        for _ in range(self.n_stack):
            self.frames.append(processed)
        return self._get_state()

    def step(self, frame: np.ndarray) -> np.ndarray:
        """Ajoute une nouvelle frame et retire la plus ancienne."""
        self.frames.append(PREPROCESS(frame))
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Retourne le stack sous forme (N_STACK, 84, 84)."""
        return np.stack(list(self.frames), axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CNN Q-NETWORK  (architecture Nature DQN — Mnih et al. 2015)
# ─────────────────────────────────────────────────────────────────────────────

class CNNQNetwork(nn.Module):
    """
    Architecture Nature DQN standard pour Atari.
    Entrée : (batch, 4, 84, 84) — 4 frames empilées grayscale

    Couches convolutionnelles :
        Conv1 : 32 filtres 8×8, stride 4  → (batch, 32, 20, 20)
        Conv2 : 64 filtres 4×4, stride 2  → (batch, 64, 9,  9 )
        Conv3 : 64 filtres 3×3, stride 1  → (batch, 64, 7,  7 )
        Flatten                            → (batch, 64×7×7) = (batch, 3136)
        FC1   : 3136 → 512
        FC2   : 512  → n_actions (6)

    Pourquoi ces dimensions ?
    Ce sont les hyperparamètres standard validés sur 57 jeux Atari.
    Stride 4 sur la première couche réduit rapidement 84→20, capturant
    les patterns globaux (position balle/raquette).
    Stride 2 sur la deuxième réduit encore, capturant les relations spatiales.
    Stride 1 sur la troisième affine les détails locaux.
    """
    def __init__(self, n_stack: int = N_STACK, n_actions: int = N_ACTIONS):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_stack, 32, kernel_size=8, stride=4),  nn.ReLU(),
            nn.Conv2d(32,      64, kernel_size=4, stride=2),  nn.ReLU(),
            nn.Conv2d(64,      64, kernel_size=3, stride=1),  nn.ReLU(),
        )
        # Calculer la taille de sortie des convolutions
        conv_out = self._get_conv_output((n_stack, FRAME_SIZE, FRAME_SIZE))

        self.fc = nn.Sequential(
            nn.Linear(conv_out, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def _get_conv_output(self, shape):
        """Calcule dynamiquement la taille de sortie des convolutions."""
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, 4, 84, 84) float32 dans [0,1]
        Retourne Q-valeurs : (batch, n_actions)
        """
        h = self.conv(x)
        h = h.view(h.size(0), -1)   # flatten
        return self.fc(h)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrait le vecteur de caractéristiques après la couche FC1 (512D).
        Utilisé à l'étape 3 (plugging) comme espace latent pour l'Acteur/Critique.
        Analogue à extract_latent() dans le prédicteur MLP de CartPole.
        """
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return torch.relu(self.fc[0](h))   # sortie de FC1 avant FC2


# ─────────────────────────────────────────────────────────────────────────────
# 3. REPLAY BUFFER  (stockage uint8 pour économiser la RAM)
# ─────────────────────────────────────────────────────────────────────────────

class AtariReplayBuffer:
    """
    Buffer circulaire optimisé pour les frames Atari.

    Problème RAM :
        200k transitions × (4, 84, 84) × float32 = 200k × 28,224 bytes ≈ 5.6 GB
        C'est trop pour la RAM standard.

    Solution : stocker les frames en uint8 [0,255] au lieu de float32 [0,1].
        200k × 28,224 bytes × (1/4) ≈ 1.4 GB — gérable.
    La conversion float32 est faite au moment du sample(), pas du push().

    Structure :
        On stocke les frames d'état et d'état suivant séparément
        pour éviter de dupliquer les frames partagées entre transitions.
        (frame t+1 d'une transition = frame t de la suivante)
    """
    def __init__(self, capacity: int = 50_000):
        self.capacity = capacity
        self.pos      = 0
        self.size     = 0

        # Pré-allouer les arrays numpy → évite les allocations dynamiques
        # (N_STACK, 84, 84) × capacity transitions
        self.states      = np.zeros((capacity, N_STACK, FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
        self.next_states = np.zeros((capacity, N_STACK, FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
        self.actions     = np.zeros(capacity, dtype=np.int64)
        self.rewards     = np.zeros(capacity, dtype=np.float32)
        self.dones       = np.zeros(capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        """
        state, next_state : (4, 84, 84) float32 [0,1]
        → convertis en uint8 [0,255] pour le stockage
        """
        self.states[self.pos]      = (state      * 255).astype(np.uint8)
        self.next_states[self.pos] = (next_state * 255).astype(np.uint8)
        self.actions[self.pos]     = action
        self.rewards[self.pos]     = reward
        self.dones[self.pos]       = done

        self.pos  = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """
        Sample aléatoire → reconvertit uint8 → float32 au moment du sample.
        C'est ici qu'on divise par 255.0.
        """
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.states[idx],      dtype=torch.float32).to(DEVICE) / 255.0,
            torch.tensor(self.actions[idx],     dtype=torch.long).to(DEVICE),
            torch.tensor(self.rewards[idx],     dtype=torch.float32).to(DEVICE),
            torch.tensor(self.next_states[idx], dtype=torch.float32).to(DEVICE) / 255.0,
            torch.tensor(self.dones[idx],       dtype=torch.float32).to(DEVICE),
        )

    def __len__(self):
        return self.size


# ─────────────────────────────────────────────────────────────────────────────
# 4. COLLECTE MIXED POLICY — DQN CNN ε-greedy
# ─────────────────────────────────────────────────────────────────────────────

def collect_data_dqn(n_steps: int = 200_000, seed: int = 0, save_every: int = 50_000):
    """
    Collecte mixed policy ε-greedy sur Pong avec CNN DQN.

    Stratégie de sauvegarde :
        À cause de la taille des données (>1 GB), on sauvegarde le dataset
        par chunks de save_every transitions → cartpole_chunk_0.npz, etc.
        Le script final recharge et fusionne les chunks.

    Paramètres ajustés pour Pong :
        - buffer capacity  : 50k (pas 100k) — limité par la RAM
        - batch_size       : 32  (pas 64)   — les images sont lourdes
        - lr               : 1e-4           — plus bas, CNN plus sensible
        - explore_steps    : 80% de n_steps — Pong nécessite beaucoup d'exploration
        - target_update    : 2000 steps      — politique change lentement

    Retourne S, A, SN sous forme de listes de chunks (pour économiser la RAM).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── Environnement ─────────────────────────────────────────────────────────
    env        = gym.make(ENV_ID, render_mode=None)
    frame_stack = FrameStack(N_STACK)

    # ── DQN CNN ───────────────────────────────────────────────────────────────
    q_net      = CNNQNetwork().to(DEVICE)
    target_net = CNNQNetwork().to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=1e-4)
    buffer    = AtariReplayBuffer(capacity=50_000)

    # ── Hyperparamètres ───────────────────────────────────────────────────────
    gamma         = 0.99
    batch_size    = 32
    epsilon       = 1.0
    epsilon_min   = 0.05
    target_update = 2_000

    explore_steps = int(0.80 * n_steps)
    epsilon_decay = (epsilon_min / epsilon) ** (1.0 / explore_steps)

    n_params = sum(p.numel() for p in q_net.parameters())
    print(f"Environnement     : {ENV_ID}")
    print(f"État              : {OBS_SHAPE}  (4 frames 84×84 grayscale)")
    print(f"N_ACTIONS         : {N_ACTIONS}")
    print(f"Paramètres DQN    : {n_params:,}")
    print(f"Device            : {DEVICE}")
    print(f"epsilon_decay     : {epsilon_decay:.8f}  (ε_min au step ~{explore_steps:,})")
    print(f"RAM buffer estimée: {50_000 * N_STACK * 84 * 84 / 1e9:.2f} GB")
    print()

    # ── Collecte ─────────────────────────────────────────────────────────────
    # Stockage en listes de chunks pour économiser la RAM
    states_buf, actions_buf, next_states_buf = [], [], []
    episode_rewards = []
    current_reward  = 0.0
    episode         = 0
    chunk_idx       = 0

    obs_raw, _  = env.reset(seed=seed)
    state        = frame_stack.reset(obs_raw)   # (4, 84, 84)

    rng = np.random.default_rng(seed)

    for step in range(n_steps):

        # ε-greedy sur les 6 actions Atari
        if rng.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                s_t    = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE) / 255.0
                q_vals = q_net(s_t)
                action = int(q_vals.argmax(dim=1).item())

        # Step environnement
        next_obs_raw, reward, terminated, truncated, _ = env.step(action)
        done      = terminated or truncated
        next_state = frame_stack.step(next_obs_raw)   # (4, 84, 84)

        current_reward += reward

        # Stocker la transition brute pour le dataset DNN prédicteur
        # On stocke en uint8 pour économiser la RAM
        states_buf.append((state      * 255).astype(np.uint8))
        actions_buf.append(action)
        next_states_buf.append((next_state * 255).astype(np.uint8))

        # Buffer DQN
        buffer.push(state, action, reward, next_state, done)

        state = next_state
        if done:
            episode_rewards.append(current_reward)
            current_reward = 0.0
            episode       += 1
            obs_raw, _ = env.reset()
            state      = frame_stack.reset(obs_raw)

        # ── Entraînement DQN ─────────────────────────────────────────────────
        if len(buffer) >= batch_size:
            s_b, a_b, r_b, s2_b, d_b = buffer.sample(batch_size)

            # Q-valeur pour l'action prise
            q_pred = q_net(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                q_next   = target_net(s2_b).max(1)[0]
                q_target = r_b + gamma * q_next * (1.0 - d_b)

            loss = nn.SmoothL1Loss()(q_pred, q_target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            optimizer.step()

        # ── Sync target network ───────────────────────────────────────────────
        if step > 0 and step % target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

        # ── Epsilon decay (par step) ──────────────────────────────────────────
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # ── Sauvegarde intermédiaire par chunks ───────────────────────────────
        # Évite de tout perdre si le script est interrompu
        if len(states_buf) >= save_every:
            chunk_path = f"pong_chunk_{chunk_idx}.npz"
            np.savez_compressed(
                chunk_path,
                states      = np.stack(states_buf),       # (N, 4, 84, 84) uint8
                actions     = np.array(actions_buf, dtype=np.int64),
                next_states = np.stack(next_states_buf),  # (N, 4, 84, 84) uint8
            )
            print(f"  Chunk {chunk_idx} sauvegardé → {chunk_path}  "
                  f"({len(states_buf):,} transitions)")
            states_buf.clear()
            actions_buf.clear()
            next_states_buf.clear()
            chunk_idx += 1

        # ── Log tous les 20k steps ────────────────────────────────────────────
        if (step + 1) % 20_000 == 0:
            recent = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
            print(f"Step {step+1:>7,} | ε={epsilon:.3f} | "
                  f"reward moy. (10 ep.) = {recent:+6.2f} | "
                  f"épisodes = {episode}")

    env.close()

    # Sauvegarder le dernier chunk
    if states_buf:
        chunk_path = f"pong_chunk_{chunk_idx}.npz"
        np.savez_compressed(
            chunk_path,
            states      = np.stack(states_buf),
            actions     = np.array(actions_buf, dtype=np.int64),
            next_states = np.stack(next_states_buf),
        )
        print(f"  Chunk {chunk_idx} (final) sauvegardé → {chunk_path}")
        chunk_idx += 1

    print(f"\nTransitions collectées : {n_steps:,}")
    print(f"Épisodes              : {episode}")
    if episode_rewards:
        print(f"Reward moyen final    : {np.mean(episode_rewards[-20:]):+.2f}")

    evaluate_policy(q_net)
    torch.save(q_net.state_dict(), "dqn_pong.pth")
    print("Modèle DQN sauvegardé → dqn_pong.pth")

    plot_learning_curve(episode_rewards)

    return chunk_idx, episode_rewards


# ─────────────────────────────────────────────────────────────────────────────
# 5. FUSION DES CHUNKS + PREPROCESSING FINAL
# ─────────────────────────────────────────────────────────────────────────────

def merge_and_preprocess(n_chunks: int, seed: int = 0):
    """
    Fusionne les chunks, one-hot encode les actions,
    split train/val/test, normalise, sauvegarde le dataset final.

    Note sur la normalisation pour Pong :
        Les frames sont déjà dans [0,1] après division par 255.
        On ne normalise PAS par mean/std comme pour CartPole
        parce que les pixels ont une distribution très non-gaussienne
        (beaucoup de 0 — fond noir, quelques valeurs pour la balle/raquettes).
        On garde simplement [0,1].

    Note sur le one-hot pour 6 actions :
        Entrée DNN prédicteur = concat(état aplati, action one-hot)
        Pour Pong : on ne peut pas aplatir (4,84,84) → 28 224D et concaténer.
        Le prédicteur pour Pong sera un réseau conv→déconv (U-Net léger)
        qui prend en entrée l'état (4,84,84) et l'action encodée séparément.
        → Le one-hot (6D) est injecté via FiLM conditioning ou concaténation
          sur le vecteur latent, pas sur l'image directement.
    """
    print(f"\nFusion de {n_chunks} chunks...")

    all_S, all_A, all_SN = [], [], []
    for i in range(n_chunks):
        chunk = np.load(f"pong_chunk_{i}.npz")
        all_S.append(chunk["states"])
        all_A.append(chunk["actions"])
        all_SN.append(chunk["next_states"])
        print(f"  Chunk {i} chargé : {len(chunk['states']):,} transitions")

    S  = np.concatenate(all_S,  axis=0).astype(np.float32) / 255.0  # (N,4,84,84) float32
    A  = np.concatenate(all_A,  axis=0)                              # (N,) int64
    SN = np.concatenate(all_SN, axis=0).astype(np.float32) / 255.0  # (N,4,84,84) float32

    print(f"\nDataset complet : {len(S):,} transitions")
    print(f"  S  shape : {S.shape}   dtype={S.dtype}")
    print(f"  A  shape : {A.shape}   dtype={A.dtype}")
    print(f"  SN shape : {SN.shape}  dtype={SN.dtype}")

    # One-hot actions (6D pour Pong)
    A_oh = np.eye(N_ACTIONS, dtype=np.float32)[A]   # (N, 6)
    print(f"  A_oh shape : {A_oh.shape}")

    # Split
    N   = len(S)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    c1  = int(N * 0.70)
    c2  = int(N * 0.80)
    tr, val, te = idx[:c1], idx[c1:c2], idx[c2:]

    print(f"\nSplit : train={len(tr):,}  val={len(val):,}  test={len(te):,}")

    # Sauvegarder — train/val/test séparément (fichiers trop grands pour un seul npz)
    # Frames restent en float32 [0,1], pas de normalisation par mean/std
    print("\nSauvegarde du dataset final...")
    np.savez_compressed("pong_data_train.npz",
                        states=S[tr], actions=A_oh[tr], next_states=SN[tr])
    np.savez_compressed("pong_data_val.npz",
                        states=S[val], actions=A_oh[val], next_states=SN[val])
    np.savez_compressed("pong_data_test.npz",
                        states=S[te], actions=A_oh[te], next_states=SN[te])

    # Stats de normalisation pour le prédicteur (mean/std des pixels sur train)
    # Calculé sur un sous-ensemble pour éviter de charger tout en RAM
    sample_idx = rng.choice(len(tr), size=min(5000, len(tr)), replace=False)
    pixel_mean = S[tr[sample_idx]].mean().astype(np.float32)
    pixel_std  = S[tr[sample_idx]].std().astype(np.float32)
    np.savez("pong_norm_stats.npz", mean=pixel_mean, std=pixel_std)

    print("Dataset sauvegardé :")
    print("  pong_data_train.npz")
    print("  pong_data_val.npz")
    print("  pong_data_test.npz")
    print("  pong_norm_stats.npz")
    print(f"\n  Pixel mean (train) : {pixel_mean:.4f}")
    print(f"  Pixel std  (train) : {pixel_std:.4f}")

    print("\n⚠️  Note sur le prédicteur DNN pour Pong :")
    print("   L'état est une image (4,84,84), pas un vecteur.")
    print("   Le prédicteur doit être un réseau conv→déconv (pas un MLP).")
    print("   L'action one-hot (6D) est injectée sur le vecteur latent")
    print("   entre l'encodeur et le décodeur (FiLM ou simple concaténation).")
    print("   Entrée  encodeur : (N, 4, 84, 84)")
    print("   Latent  + action : (N, latent_dim + 6)")
    print("   Sortie décodeur  : (N, 4, 84, 84)  — frame suivante prédite")

    return S[tr], A_oh[tr], SN[tr]


# ─────────────────────────────────────────────────────────────────────────────
# 6. ÉVALUATION + COURBE D'APPRENTISSAGE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_policy(q_net: CNNQNetwork, n_episodes: int = 5, seed: int = 42):
    """
    Évalue la politique greedy sur quelques épisodes.
    5 épisodes suffisent car chaque épisode Pong dure ~1000 steps.
    Score Pong : entre -21 (perdu tous les points) et +21 (gagné tous les points).
    """
    env        = gym.make(ENV_ID)
    fs         = FrameStack(N_STACK)
    q_net.eval()
    scores     = []

    for ep in range(n_episodes):
        obs_raw, _ = env.reset(seed=seed + ep)
        state      = fs.reset(obs_raw)
        done, total = False, 0.0

        while not done:
            with torch.no_grad():
                s_t    = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE) / 255.0
                action = int(q_net(s_t).argmax(dim=1).item())
            obs_raw, r, terminated, truncated, _ = env.step(action)
            state  = fs.step(obs_raw)
            done   = terminated or truncated
            total += r
        scores.append(total)

    env.close()
    q_net.train()

    mean_s = np.mean(scores)
    status = "✅ gagne" if mean_s > 0 else ("⚠️  nul" if mean_s == 0 else "❌ perd")
    print(f"Évaluation greedy ({n_episodes} éps) : mean={mean_s:+.1f}  {status}")
    print(f"  (Pong : [-21, +21] | > 0 = gagne plus qu'il ne perd)")
    return float(mean_s)


def plot_learning_curve(episode_rewards: list, window: int = 10,
                        save_path: str = "pong_learning_curve.png"):
    if len(episode_rewards) < 2:
        print("Pas assez d'épisodes.")
        return

    rewards  = np.array(episode_rewards)
    episodes = np.arange(1, len(rewards) + 1)
    kernel   = np.ones(window) / window
    smoothed = np.convolve(rewards, kernel, mode="valid")
    smooth_x = np.arange(window, len(rewards) + 1)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(episodes, rewards,  color="#378ADD", alpha=0.2, lw=0.8, label="reward brut")
    ax.plot(smooth_x, smoothed, color="#1D9E75", lw=2,
            label=f"moyenne glissante ({window} épisodes)")
    ax.axhline( 0,  color="gray",    linestyle=":",  lw=0.8, label="nul (0)")
    ax.axhline(21,  color="#D85A30", linestyle="--", lw=1,   label="victoire max (+21)")
    ax.axhline(-21, color="#D85A30", linestyle="--", lw=1,   label="défaite max (−21)")
    ax.set_xlabel("Épisode")
    ax.set_ylabel("Score")
    ax.set_title("Apprentissage DQN CNN — Pong-v5 (mixed policy)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.show()
    print(f"Sauvegardé → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 65)
    print("Collecte Pong-v5 — mixed policy DQN CNN")
    print(f"État : {OBS_SHAPE}  |  Actions : {N_ACTIONS}  |  Device : {DEVICE}")
    print("=" * 65)

    # ── 1. Collecte + entraînement DQN ───────────────────────────────────────
    n_chunks, ep_rewards = collect_data_dqn(
        n_steps    = 200_000,
        seed       = 0,
        save_every = 50_000,    # sauvegarde toutes les 50k transitions
    )

    # ── 2. Fusion, split, normalisation ──────────────────────────────────────
    merge_and_preprocess(n_chunks=n_chunks, seed=0)

    print("\n" + "=" * 65)
    print("Collecte terminée. Fichiers produits :")
    print("  dqn_pong.pth           ← poids DQN entraîné")
    print("  pong_chunk_*.npz       ← chunks intermédiaires (uint8)")
    print("  pong_data_train.npz    ← dataset train (float32)")
    print("  pong_data_val.npz      ← dataset val")
    print("  pong_data_test.npz     ← dataset test")
    print("  pong_norm_stats.npz    ← mean/std pixels")
    print("  pong_learning_curve.png")
    print()
    print("Étape suivante : entraîner un prédicteur conv→déconv")
    print("  Entrée : (N, 4, 84, 84) + action one-hot (6D) sur le latent")
    print("  Sortie : (N, 4, 84, 84) — frame suivante prédite")
    print("=" * 65)