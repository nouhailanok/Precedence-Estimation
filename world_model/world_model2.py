"""
world_model.py
==============
Prediction network avec 3 configurations :

  Config 1 — Standard   : prédit S(t+1) directement
                          entrée : [St | At_oh]   sortie : S(t+1)

  Config 2 — MultiStep  : prédit S(t+n) directement
                          entrée : [St | At_oh | At+1_oh | ... | At+n-1_oh]
                          sortie : S(t+n)
                          → inférence via Solution B :
                            WM_C3 génère les états intermédiaires,
                            QNet choisit les actions futures

  Config 3 — Delta      : prédit ΔS = S(t+1) - St
                          entrée : [St | At_oh]   sortie : ΔS
                          S(t+1) = St + ΔS

IMPORTANT — Config 2 et données séquentielles :
  data_collector.py sauvegarde des splits MÉLANGÉS (shuffle aléatoire).
  Config 2 a besoin de transitions CONSÉCUTIVES (ordre temporel préservé).
  → train_world_model collecte automatiquement des données séquentielles
    pour Config 2, en réutilisant mean/std du .npz pour cohérence.

Workflow recommandé :
  # 1. Collecter une fois (splits shufflés pour configs 1 & 3)
  python collect_data/data_collect_general.py --env CartPole-v1

  # 2. Entraîner chaque config
  python world_model/world_model2.py --env CartPole-v1 --config 1 --data collect_data/data/data_CartPole-v1.npz
  python world_model/world_model2.py --env CartPole-v1 --config 2 --data collect_data/data/data_CartPole-v1.npz
  python world_model/world_model2.py --env CartPole-v1 --config 3 --data collect_data/data/data_CartPole-v1.npz

  python world_model/world_model2.py --env MountainCar-v0 --config 1 --data collect_data/data/data_MountainCar-v0.npz
  python world_model/world_model2.py --env MountainCar-v0 --config 2 --data collect_data/data/data_MountainCar-v0.npz
  python world_model/world_model2.py --env MountainCar-v0 --config 3 --data collect_data/data/data_MountainCar-v0.npz


  python world_model/world_model2.py --env LunarLander-v3 --config 1 --data collect_data/data/data_LunarLander-v3.npz
  python world_model/world_model2.py --env LunarLander-v3 --config 2 --data collect_data/data/data_LunarLander-v3.npz
  python world_model/world_model2.py --env LunarLander-v3 --config 3 --data collect_data/data/data_LunarLander-v3.npz

  # Ou les 3 d'un coup
  python world_model/world_model2.py --env CartPole-v1 --all --data collect_data/data/data_CartPole-v1.npz

Import dans un agent :
  from world_model import WorldModelWrapper
  wm = WorldModelWrapper.load("checkpoints/wm_CartPole-v1_config3.pth")
  s_next = wm.predict(state, action)
"""

import argparse
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────
#  Chemins
# ─────────────────────────────────────────────
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = BASE_DIR / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
#  Hyperparamètres
# ─────────────────────────────────────────────
COLLECT_STEPS = 30_000
EPOCHS        = 200
BATCH_SIZE    = 256
LR            = 1e-3
HIDDEN        = 64
SEED          = 42


# ════════════════════════════════════════════════════════════
#  1. ARCHITECTURE
# ════════════════════════════════════════════════════════════

class WorldModel(nn.Module):
    """
    Réseau de prédiction d'état (3 configs).

    Config 1 : in = obs_dim + act_dim          → S(t+1) absolu
    Config 2 : in = obs_dim + n_step * act_dim → S(t+n) absolu
    Config 3 : in = obs_dim + act_dim          → ΔS  (S+ΔS dans predict())
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 config: int = 3, n_step: int = 3,
                 hidden: int = HIDDEN):
        super().__init__()
        assert config in (1, 2, 3), "config doit être 1, 2 ou 3"

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.config  = config
        self.n_step  = n_step if config == 2 else 1

        in_dim = obs_dim + (n_step * act_dim if config == 2 else act_dim)

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, obs_dim),
        )

    def forward(self, state: torch.Tensor,
                action_input: torch.Tensor) -> torch.Tensor:
        """
        state        : (batch, obs_dim)
        action_input : (batch, act_dim)              configs 1 & 3
                     | (batch, n_step * act_dim)     config  2
        retourne     : (batch, obs_dim)
        """
        return self.net(torch.cat([state, action_input], dim=-1))

    def predict(self, state: torch.Tensor,
                action_input: torch.Tensor) -> torch.Tensor:
        """Retourne toujours un état absolu (applique S+ΔS pour config 3)."""
        out = self.forward(state, action_input)
        return state + out if self.config == 3 else out


# ════════════════════════════════════════════════════════════
#  2. WRAPPER (interface agents)
# ════════════════════════════════════════════════════════════

class WorldModelWrapper:
    """
    Encapsule WorldModel + normalisation + règles physiques par env.

    Configs 1 & 3 :
        s_next = wrapper.predict(state, action)

    Config 2 :
        s_tn, actions = wrapper.predict_multistep(state, qnet, wm_c3, epsilon)
    """

    ENV_BOUNDS = {
        "CartPole-v1": {
            "low":  np.array([-4.8, -5.0, -0.418, -5.0], dtype=np.float32),
            "high": np.array([ 4.8,  5.0,  0.418,  5.0], dtype=np.float32),
        },
        "MountainCar-v0": {
            "low":  np.array([-1.2, -0.07], dtype=np.float32),
            "high": np.array([ 0.6,  0.07], dtype=np.float32),
        },
        "LunarLander-v2": {
            "low":  np.full(8, -np.inf, dtype=np.float32),
            "high": np.full(8,  np.inf, dtype=np.float32),
        },
        "LunarLander-v3": {
            "low":  np.full(8, -np.inf, dtype=np.float32),
            "high": np.full(8,  np.inf, dtype=np.float32),
        },
    }

    def __init__(self, model: WorldModel, mean: np.ndarray,
                 std: np.ndarray, env_name: str = "",
                 device: torch.device = None):
        self.model    = model
        self.mean     = mean.astype(np.float32)
        self.std      = std.astype(np.float32)
        self.env_name = env_name
        self.device   = device or torch.device("cpu")

        bounds    = self.ENV_BOUNDS.get(env_name)
        self.low  = bounds["low"]  if bounds else None
        self.high = bounds["high"] if bounds else None

        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    # ── Normalisation ────────────────────────────────────────

    def _norm(self, s: np.ndarray) -> np.ndarray:
        return (s.astype(np.float32) - self.mean) / (self.std + 1e-8)

    def _denorm(self, s_norm: np.ndarray) -> np.ndarray:
        return s_norm * (self.std + 1e-8) + self.mean

    def _clip(self, s: np.ndarray) -> np.ndarray:
        return np.clip(s, self.low, self.high) if self.low is not None else s

    def _onehot(self, a: int) -> np.ndarray:
        oh = np.zeros(self.model.act_dim, dtype=np.float32)
        oh[a] = 1.0
        return oh

    # ── Prédiction Config 1 & 3 ──────────────────────────────

    def predict(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Prédit S(t+1) — configs 1 & 3 uniquement.
        state  : np.ndarray (obs_dim,)
        action : int
        """
        assert self.model.config in (1, 3), (
            "predict() est pour configs 1 & 3. "
            "Utilise predict_multistep() pour config 2."
        )
        s_t = torch.FloatTensor(self._norm(state)).unsqueeze(0).to(self.device)
        a_t = torch.FloatTensor(self._onehot(action)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            s_next_norm = self.model.predict(s_t, a_t).cpu().numpy()[0]

        return self._clip(self._denorm(s_next_norm))

    # ── Prédiction Config 2 — Solution B ─────────────────────

    def predict_multistep(self, state: np.ndarray,
                          qnet: nn.Module,
                          wm_c3: "WorldModelWrapper",
                          epsilon: float = 0.0) -> tuple:
        """
        Prédit S(t+n) via Solution B :
          1. Générer les n actions futures : QNet sur états simulés par WM_C3
          2. Prédiction directe à n steps avec WM Config2

        Retourne :
          s_tn          : np.ndarray (obs_dim,)
          future_actions: list[int] longueur n_step
        """
        assert self.model.config == 2, "predict_multistep() → config 2 uniquement."
        assert wm_c3.model.config == 3, "wm_c3 doit être config 3."

        n_step      = self.model.n_step
        act_dim     = self.model.act_dim
        future_acts = []
        s_sim       = state.copy()

        # Étape 1 : générer les n actions futures
        for k in range(n_step):
            if np.random.rand() < epsilon:
                a_k = np.random.randint(0, act_dim)
            else:
                s_t = torch.FloatTensor(s_sim).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    a_k = int(qnet(s_t).argmax().item())
            future_acts.append(a_k)
            if k < n_step - 1:
                s_sim = wm_c3.predict(s_sim, a_k)

        # Étape 2 : prédiction directe à n steps
        s_norm = self._norm(state)
        a_vec  = np.concatenate([self._onehot(a) for a in future_acts])
        s_t    = torch.FloatTensor(s_norm).unsqueeze(0).to(self.device)
        a_t    = torch.FloatTensor(a_vec).unsqueeze(0).to(self.device)

        with torch.no_grad():
            s_tn_norm = self.model.predict(s_t, a_t).cpu().numpy()[0]

        return self._clip(self._denorm(s_tn_norm)), future_acts

    # ── Règles physiques ─────────────────────────────────────

    def is_done(self, s_next: np.ndarray) -> bool:
        env = self.env_name
        if env == "CartPole-v1":
            return bool(abs(s_next[0]) > 2.4 or abs(s_next[2]) > 0.209)
        if env == "MountainCar-v0":
            return bool(s_next[0] >= 0.5)
        if env in ("LunarLander-v2", "LunarLander-v3"):
            return bool(s_next[1] <= 0.0)
        return False

    def synthetic_reward(self, s_next: np.ndarray, done: bool) -> float:
        env = self.env_name
        if env == "CartPole-v1":
            return 0.0 if done else 1.0
        if env == "MountainCar-v0":
            return 0.0 if done else -1.0
        if env in ("LunarLander-v2", "LunarLander-v3"):
            if done:
                return 100.0 if s_next[1] >= 0.0 else -100.0
            return -0.3
        return 1.0

    # ── Sauvegarde / Chargement ──────────────────────────────

    def save(self, path: str):
        torch.save({
            "state_dict": self.model.state_dict(),
            "obs_dim":    self.model.obs_dim,
            "act_dim":    self.model.act_dim,
            "config":     self.model.config,
            "n_step":     self.model.n_step,
            "hidden":     HIDDEN,
            "mean":       self.mean,
            "std":        self.std,
            "env_name":   self.env_name,
        }, path)
        print(f"[WM] Sauvegardé → {path}")

    @classmethod
    def load(cls, path: str,
             device: torch.device = None) -> "WorldModelWrapper":
        device = device or torch.device("cpu")
        ckpt   = torch.load(path, map_location=device)
        model  = WorldModel(
            obs_dim = ckpt["obs_dim"],
            act_dim = ckpt["act_dim"],
            config  = ckpt["config"],
            n_step  = ckpt["n_step"],
            hidden  = ckpt.get("hidden", HIDDEN),
        )
        model.load_state_dict(ckpt["state_dict"])
        wrapper = cls(
            model    = model,
            mean     = ckpt["mean"],
            std      = ckpt["std"],
            env_name = ckpt.get("env_name", ""),
            device   = device,
        )
        print(f"[WM] Chargé ← {path}  "
              f"(env={wrapper.env_name}, config={model.config}, "
              f"n_step={model.n_step})")
        return wrapper


# ════════════════════════════════════════════════════════════
#  3. CONSTRUCTION DU DATASET
# ════════════════════════════════════════════════════════════

def build_dataset(data: dict, config: int, n_step: int,
                  device: torch.device,
                  split: str = "train") -> tuple:
    """
    Construit (X, Y) depuis le dict produit par data_collector.py.

    data   : dict avec clés states_train/val/test, mean, std, ...
    config : 1, 2 ou 3
    split  : "train" | "val" | "test"

    Config 1 : X=[s_norm|a_oh]       Y=s2_norm
    Config 3 : X=[s_norm|a_oh]       Y=s2_norm - s_norm  (ΔS normalisé)
    Config 2 : → géré par _build_multistep (données séquentielles requises)
    """
    assert config in (1, 3), (
        f"build_dataset() est pour configs 1 & 3. "
        f"Config 2 utilise _build_multistep()."
    )
    assert split in ("train", "val", "test"), \
        f"split doit être 'train', 'val' ou 'test', reçu : {split}"

    mean    = data["mean"].astype(np.float32)
    std     = (data["std"] + 1e-8).astype(np.float32)

    states  = data[f"states_{split}"].astype(np.float32)
    next_s  = data[f"next_states_{split}"].astype(np.float32)
    actions = data[f"actions_{split}"].astype(np.int32)
    act_dim = int(actions.max()) + 1

    s_norm  = (states - mean) / std
    s2_norm = (next_s - mean) / std
    a_oh    = np.eye(act_dim, dtype=np.float32)[actions]

    X = np.concatenate([s_norm, a_oh], axis=1)
    Y = s2_norm if config == 1 else (s2_norm - s_norm)

    print(f"[DATA] Config {config} | {split:5s} : {X.shape[0]:,} samples")
    return (torch.FloatTensor(X).to(device),
            torch.FloatTensor(Y).to(device))


def _build_multistep(states: np.ndarray,
                     next_states: np.ndarray,
                     actions: np.ndarray,
                     dones: np.ndarray,
                     n_step: int,
                     mean: np.ndarray,
                     std: np.ndarray,
                     device: torch.device) -> tuple:
    """
    Config 2 — construit (S0, [A0..An-1]) → S(t+n) sur données SÉQUENTIELLES.

    Reçoit directement les arrays numpy (pas le dict splitté).
    Rejette toute séquence qui traverse un épisode terminé.

    Exemple n_step=3, i=5 :
      Valide   si done[5]=F et done[6]=F → X=[s5|a5_oh|a6_oh|a7_oh], Y=s8
      Invalide si done[5]=T ou done[6]=T → rejeté
    """
    N       = len(states)
    act_dim = int(actions.max()) + 1
    mean    = mean.astype(np.float32)
    std     = (std + 1e-8).astype(np.float32)

    X_list, Y_list = [], []
    rejected = 0

    for i in range(N - n_step):
        # Rejeter si done apparaît dans les n-1 premiers steps
        if np.any(dones[i : i + n_step - 1] > 0.5):
            rejected += 1
            continue

        s0_norm  = (states[i]                       - mean) / std
        stn_norm = (next_states[i + n_step - 1]     - mean) / std

        a_seq = np.zeros(n_step * act_dim, dtype=np.float32)
        for k in range(n_step):
            a_seq[k * act_dim + int(actions[i + k])] = 1.0

        X_list.append(np.concatenate([s0_norm, a_seq]))
        Y_list.append(stn_norm)

    kept     = len(X_list)
    total    = N - n_step
    pct_kept = 100 * kept / max(total, 1)

    print(f"[DATA] Config 2 (n={n_step}) : "
          f"{kept:,} séquences valides / {total:,} "
          f"({pct_kept:.1f}% conservées, {rejected:,} rejetées)")

    if kept == 0:
        raise ValueError(
            f"Aucune séquence multi-step valide (n_step={n_step}). "
            "Les épisodes sont trop courts ou n_step trop grand. "
            "Essaie --nstep 2 ou augmente --collect."
        )

    X = torch.FloatTensor(np.array(X_list)).to(device)
    Y = torch.FloatTensor(np.array(Y_list)).to(device)
    return X, Y


# ════════════════════════════════════════════════════════════
#  4. EARLY STOPPING
# ════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 1e-6):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = float("inf")
        self.num_bad   = 0

    def step(self, loss: float) -> bool:
        """Retourne True si on doit arrêter."""
        if loss < self.best - self.min_delta:
            self.best    = loss
            self.num_bad = 0
            return False
        self.num_bad += 1
        return self.num_bad >= self.patience


# ════════════════════════════════════════════════════════════
#  5. COLLECTE INTERNE (fallback + config 2)
# ════════════════════════════════════════════════════════════

def _collect_sequential(env_name: str, n_steps: int,
                        seed: int) -> tuple:
    """
    Collecte des transitions SÉQUENTIELLES (ordre temporel préservé).
    Utilisé pour :
      - Config 2 (multistep) qui a besoin de séquences consécutives
      - Fallback si data_collector.py est absent

    Retourne : (states, next_states, actions, dones) — arrays numpy séquentiels
    """
    print(f"[COLLECT] Collecte séquentielle pour Config 2 "
          f"({n_steps} steps, env={env_name})")

    env    = gym.make(env_name)
    obs, _ = env.reset(seed=seed)
    N      = n_steps
    obs_d  = env.observation_space.shape[0]

    states      = np.zeros((N, obs_d), dtype=np.float32)
    next_states = np.zeros((N, obs_d), dtype=np.float32)
    actions     = np.zeros(N,          dtype=np.int32)
    dones       = np.zeros(N,          dtype=np.float32)

    for i in range(N):
        a = env.action_space.sample()
        ns, _, term, trunc, _ = env.step(a)
        done = term or trunc
        states[i]      = obs
        actions[i]     = a
        next_states[i] = ns
        dones[i]       = float(done)
        obs = env.reset()[0] if done else ns

    env.close()
    return states, next_states, actions, dones


# ════════════════════════════════════════════════════════════
#  6. ENTRAÎNEMENT
# ════════════════════════════════════════════════════════════

def train_world_model(env_name: str, config: int = 3,
                      n_step: int = 3, hidden: int = HIDDEN,
                      epochs: int = EPOCHS, seed: int = SEED,
                      collect_steps: int = COLLECT_STEPS,
                      data_path: str = None) -> WorldModelWrapper:
    """
    Pipeline complet :
      1. Charger le .npz (data_collector.py) ou collecter en fallback
      2. Construire le dataset selon la config
      3. Entraîner avec early stopping sur val_loss
      4. Évaluer sur test set et sauvegarder

    data_path : chemin .npz produit par data_collector.py
                Si None → collecte automatique
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  WorldModel | env={env_name} | config={config}"
          + (f" | n_step={n_step}" if config == 2 else ""))
    print(f"  device={device} | hidden={hidden} | epochs={epochs}")
    print(f"{'='*60}\n")

    # ── Dimensions ───────────────────────────────────────────
    env     = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    env.close()

    # ── Chargement du .npz ───────────────────────────────────
    data = None
    if data_path and os.path.exists(data_path):
        try:
            from collect_data.data_collect_general import load_dataset
            data = load_dataset(data_path)
            print(f"[DATA] Chargé via data_collect_general.load_dataset()")
        except ImportError:
            npz  = np.load(data_path, allow_pickle=True)
            data = {k: npz[k] for k in npz.files}
            data["env_name"] = str(data.get("env_name",
                                   np.array([env_name]))[0])
            # Assurer les bons types
            data["mean"] = data["mean"].astype(np.float32)
            data["std"]  = data["std"].astype(np.float32)
            print(f"[DATA] Chargé via numpy.load()")
    elif data_path:
        print(f"[WARN] {data_path} introuvable — collecte automatique.")

    # ── Construction du dataset selon la config ──────────────

    if config == 2:
        # Config 2 : données séquentielles obligatoires
        # Le .npz contient des splits SHUFFLÉS → on collecte des données fraîches
        # On réutilise mean/std du .npz pour rester cohérent avec configs 1 & 3
        mean_ref = data["mean"] if data else None
        std_ref  = data["std"]  if data else None

        print(f"[CONFIG2] Collecte de données séquentielles fraîches "
              f"(splits shufflés du .npz incompatibles avec multistep).")

        s_seq, ns_seq, a_seq, d_seq = _collect_sequential(
            env_name, collect_steps, seed)

        # Normalisation : utiliser mean/std du .npz si disponible,
        # sinon calculer sur les nouvelles données
        if mean_ref is None:
            mean_ref = s_seq.mean(axis=0).astype(np.float32)
            std_ref  = s_seq.std(axis=0).astype(np.float32)

        X_all, Y_all = _build_multistep(
            s_seq, ns_seq, a_seq, d_seq,
            n_step, mean_ref, std_ref, device)

        # Split train/val/test (70/15/15) sur les séquences
        N    = len(X_all)
        i_v  = int(0.70 * N)
        i_t  = int(0.85 * N)
        X_train, Y_train = X_all[:i_v],  Y_all[:i_v]
        X_val,   Y_val   = X_all[i_v:i_t], Y_all[i_v:i_t]
        X_test,  Y_test  = X_all[i_t:],  Y_all[i_t:]

        mean_save = mean_ref
        std_save  = std_ref

    else:
        # Configs 1 & 3 : utiliser les splits shufflés du .npz
        if data is None:
            # Fallback : collecte et split manuel
            s, ns, a, d = _collect_sequential(env_name, collect_steps, seed)
            N    = len(s)
            i_v  = int(0.70 * N)
            i_t  = int(0.85 * N)
            mean_save = s[:i_v].mean(axis=0).astype(np.float32)
            std_save  = s[:i_v].std(axis=0).astype(np.float32)

            data = {
                "states_train":      s[:i_v],
                "next_states_train": ns[:i_v],
                "actions_train":     a[:i_v],
                "states_val":        s[i_v:i_t],
                "next_states_val":   ns[i_v:i_t],
                "actions_val":       a[i_v:i_t],
                "states_test":       s[i_t:],
                "next_states_test":  ns[i_t:],
                "actions_test":      a[i_t:],
                "mean": mean_save,
                "std":  std_save,
            }
        else:
            mean_save = data["mean"]
            std_save  = data["std"]

        print(f"[NORM] mean = {np.round(mean_save, 4)}")
        print(f"[NORM] std  = {np.round(std_save,  4)}\n")

        X_train, Y_train = build_dataset(data, config, n_step, device, "train")
        X_val,   Y_val   = build_dataset(data, config, n_step, device, "val")
        X_test,  Y_test  = build_dataset(data, config, n_step, device, "test")

    print(f"\n[WM] train={len(X_train):,} | val={len(X_val):,} | "
          f"test={len(X_test):,}\n")

    train_loader = DataLoader(TensorDataset(X_train, Y_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val,   Y_val),
                              batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_test,  Y_test),
                              batch_size=BATCH_SIZE, shuffle=False)

    # ── Modèle ───────────────────────────────────────────────
    model   = WorldModel(obs_dim, act_dim, config, n_step, hidden).to(device)
    opt     = optim.Adam(model.parameters(), lr=LR)
    sched   = optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.5)
    loss_fn = nn.MSELoss()
    stopper = EarlyStopping(patience=8)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[WM] Paramètres : {n_params:,}\n")

    # ── Boucle d'entraînement ────────────────────────────────
    best_val = float("inf")
    best_sd  = None
    tr_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):

        # — Train —
        model.train()
        train_loss = 0.0                    # ← remis à zéro à chaque epoch
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            s_b = xb[:, :obs_dim]
            a_b = xb[:, obs_dim:]
            pred = model.forward(s_b, a_b)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_loader.dataset)
        tr_losses.append(train_loss)

        # — Validation —
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                s_b = xb[:, :obs_dim]
                a_b = xb[:, obs_dim:]
                pred = model.forward(s_b, a_b)
                val_loss += loss_fn(pred, yb).item() * len(xb)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        sched.step()

        # Checkpoint sur meilleur val_loss
        if val_loss < best_val:
            best_val = val_loss
            best_sd  = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"train={train_loss:.6f} | "
                  f"val={val_loss:.6f} | "
                  f"best_val={best_val:.6f} | "
                  f"lr={sched.get_last_lr()[0]:.1e}")

        # Early stopping sur val_loss
        if stopper.step(val_loss):
            print(f"\n[WM] Early stopping à l'epoch {epoch} "
                  f"(best_val={best_val:.6f})")
            break

    # ── Restaurer meilleur checkpoint ────────────────────────
    model.load_state_dict(best_sd)
    model.eval()

    # ── Évaluation sur test set ──────────────────────────────
    test_mse, test_mae = _evaluate_test(model, test_loader, obs_dim, device)
    print(f"[TEST] MSE={test_mse:.6f} | MAE={test_mae:.6f}")

    # ── Évaluation qualitative configs 1 & 3 ─────────────────
    if config in (1, 3) and data is not None:
        _quick_eval(WorldModelWrapper(model, mean_save, std_save,
                                      env_name, device),
                    data, split="test")

    # ── Sauvegarde ───────────────────────────────────────────
    tag      = f"config{config}" + (f"_n{n_step}" if config == 2 else "")
    savepath = str(CKPT_DIR / f"wm_{env_name}_{tag}.pth")
    wrapper  = WorldModelWrapper(model, mean_save, std_save, env_name, device)
    wrapper.save(savepath)

    return wrapper


# ════════════════════════════════════════════════════════════
#  7. ÉVALUATION
# ════════════════════════════════════════════════════════════

def _evaluate_test(model: WorldModel, test_loader: DataLoader,
                   obs_dim: int, device: torch.device) -> tuple:
    """MSE et MAE sur le test loader."""
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            s_b = xb[:, :obs_dim]
            a_b = xb[:, obs_dim:]
            p   = model.forward(s_b, a_b).cpu().numpy()
            preds.append(p)
            targets.append(yb.numpy())

    pred   = np.concatenate(preds)
    y_true = np.concatenate(targets)
    mse    = float(np.mean((pred - y_true) ** 2))
    mae    = float(np.mean(np.abs(pred - y_true)))
    return mse, mae


def _quick_eval(wrapper: WorldModelWrapper, data: dict,
                split: str = "test", n_eval: int = 500):
    """
    MAE en espace original (dénormalisé) sur n_eval transitions.
    Donne une erreur interprétable (même unité que l'état).
    """
    states_key  = f"states_{split}"
    actions_key = f"actions_{split}"
    next_key    = f"next_states_{split}"

    N     = len(data[states_key])
    n_eval = min(n_eval, N)
    idx   = np.random.choice(N, size=n_eval, replace=False)

    errors = [
        np.abs(
            wrapper.predict(data[states_key][i],
                            int(data[actions_key][i]))
            - data[next_key][i]
        ).mean()
        for i in idx
    ]
    print(f"[EVAL] MAE (espace original, {split}) : "
          f"{np.mean(errors):.5f} ± {np.std(errors):.5f}  "
          f"max={np.max(errors):.5f}\n")


# ════════════════════════════════════════════════════════════
#  8. MAIN
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Entraîne un WorldModel (config 1/2/3) sur un env Gymnasium")
    parser.add_argument("--env",     default="CartPole-v1",
                        choices=["CartPole-v1", "MountainCar-v0",
                                 "LunarLander-v2", "LunarLander-v3",
                                 "Acrobot-v1"])
    parser.add_argument("--config",  type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--nstep",   type=int, default=3,
                        help="Horizon n pour config 2 (défaut : 3)")
    parser.add_argument("--hidden",  type=int, default=HIDDEN)
    parser.add_argument("--epochs",  type=int, default=EPOCHS)
    parser.add_argument("--seed",    type=int, default=SEED)
    parser.add_argument("--data",    default=None,
                        help="Chemin .npz de data_collector.py "
                             "(ex: collect_data/data/data_CartPole-v1.npz)")
    parser.add_argument("--collect", type=int, default=COLLECT_STEPS,
                        help="Steps collectés si --data absent ou config 2 "
                             "(défaut: 30000)")
    parser.add_argument("--all",     action="store_true",
                        help="Entraîne les 3 configs (même --data pour toutes)")
    args = parser.parse_args()

    configs = [1, 2, 3] if args.all else [args.config]

    for cfg in configs:
        train_world_model(
            env_name      = args.env,
            config        = cfg,
            n_step        = args.nstep,
            hidden        = args.hidden,
            epochs        = args.epochs,
            seed          = args.seed,
            collect_steps = args.collect,
            data_path     = args.data,
        )


if __name__ == "__main__":
    main()