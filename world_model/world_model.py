"""
world_model.py
==============
Prediction network avec 3 configurations :

  Config 1 — Standard   : prédit S(t+1) directement
                          entrée : [St | At_oh]   sortie : S(t+1)

  Config 2 — MultiStep  : prédit S(t+n) directement
                          entrée : [St | At_oh | At+1_oh | ... | At+n-1_oh]
                          sortie : S(t+n)
                          → les n actions futures sont générées via Solution B
                            (WM_C3 comme simulateur intermédiaire + politique QNet)

  Config 3 — Delta      : prédit ΔS = S(t+1) - St
                          entrée : [St | At_oh]   sortie : ΔS
                          S(t+1) = St + ΔS

Workflow recommandé :
  # 1. Collecter les données une seule fois
  python data_collector.py --env CartPole-v1

  # 2. Entraîner les 3 configs sur les mêmes données
  python ./world_model/world_model.py --env CartPole-v1 --config 1 --data collect_data/data/data_CartPole-v1.npz
  python ./world_model/world_model.py --env CartPole-v1 --config 2 --data collect_data/data/data_CartPole-v1.npz
  python ./world_model/world_model.py --env CartPole-v1 --config 3 --data collect_data/data/data_CartPole-v1.npz
  

  # Ou les 3 d'un coup (même données, collecte unique)
  python ./world_model/world_model.py --env CartPole-v1 --all --data ../collect_data/data/data_CartPole-v1.npz

Usage dans un agent :
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
EPOCHS        = 100
BATCH_SIZE    = 256
LR            = 1e-3
HIDDEN        = 64
SEED          = 42


# ════════════════════════════════════════════════════════════
#  1. ARCHITECTURE DU RÉSEAU
# ════════════════════════════════════════════════════════════

class WorldModel(nn.Module):
    """
    Réseau de prédiction d'état.

    Config 1 — Standard
        in_dim  = obs_dim + act_dim
        out_dim = obs_dim         → S(t+1) directement

    Config 2 — MultiStep
        in_dim  = obs_dim + n_step * act_dim
        out_dim = obs_dim         → S(t+n) directement

    Config 3 — Delta
        in_dim  = obs_dim + act_dim
        out_dim = obs_dim         → ΔS  (S(t+1) = St + ΔS dans predict())
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
        retourne     : (batch, obs_dim)  — ΔS (config 3) ou S' (configs 1 & 2)
        """
        return self.net(torch.cat([state, action_input], dim=-1))

    def predict(self, state: torch.Tensor,
                action_input: torch.Tensor) -> torch.Tensor:
        """
        Applique S + ΔS pour config 3.
        Retourne toujours un état absolu.
        """
        out = self.forward(state, action_input)
        return state + out if self.config == 3 else out


# ════════════════════════════════════════════════════════════
#  2. WRAPPER  (interface agents)
# ════════════════════════════════════════════════════════════

class WorldModelWrapper:
    """
    Encapsule WorldModel + normalisation + règles physiques par env.
    C'est cette classe que DQN/PPO importent.

    Configs 1 & 3 :
        s_next = wrapper.predict(state, action)

    Config 2 :
        # Solution B — fournir les actions futures générées par QNet + WM_C3
        s_tn, actions = wrapper.predict_multistep(state, qnet, wm_c3, epsilon)
    """

    # Bornes physiques par environnement (pour clipping des états prédits)
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
            p.requires_grad = False   # toujours gelé à l'usage

    # ── Normalisation ────────────────────────────────────────

    def _norm(self, s: np.ndarray) -> np.ndarray:
        return (s.astype(np.float32) - self.mean) / (self.std + 1e-8)

    def _denorm(self, s_norm: np.ndarray) -> np.ndarray:
        return s_norm * (self.std + 1e-8) + self.mean

    def _clip(self, s: np.ndarray) -> np.ndarray:
        if self.low is not None:
            return np.clip(s, self.low, self.high)
        return s

    def _onehot(self, a: int) -> np.ndarray:
        oh = np.zeros(self.model.act_dim, dtype=np.float32)
        oh[a] = 1.0
        return oh

    # ── Prédiction Config 1 & 3 ──────────────────────────────

    def predict(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Prédit S(t+1) pour configs 1 et 3.
        state  : np.ndarray (obs_dim,)
        action : int
        retourne : np.ndarray (obs_dim,)
        """
        assert self.model.config in (1, 3), (
            "predict() est pour configs 1 & 3. "
            "Utilise predict_multistep() pour config 2."
        )
        s_norm = self._norm(state)
        a_oh   = self._onehot(action)

        s_t = torch.FloatTensor(s_norm).unsqueeze(0).to(self.device)
        a_t = torch.FloatTensor(a_oh).unsqueeze(0).to(self.device)

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
          1. Générer les n actions futures avec QNet + WM_C3 (simulateur 1-step)
          2. Donner [St, A0..An-1] au WM Config2 → S(t+n) direct

        Paramètres :
          state   : np.ndarray (obs_dim,)
          qnet    : réseau Q de l'agent (nn.Module)
          wm_c3   : WorldModelWrapper config 3 (simulateur intermédiaire)
          epsilon : taux d'exploration pour la politique ε-greedy

        Retourne :
          s_tn          : np.ndarray (obs_dim,) — état prédit à t+n
          future_actions: list[int] longueur n_step — actions utilisées
        """
        assert self.model.config == 2, "predict_multistep() est pour config 2 uniquement."
        assert wm_c3.model.config == 3, "wm_c3 doit être un WorldModelWrapper config 3."

        n_step      = self.model.n_step
        act_dim     = self.model.act_dim
        future_acts = []
        s_sim       = state.copy()

        # ── Étape 1 : générer les n actions futures ──────────
        for k in range(n_step):
            # Politique ε-greedy sur l'état simulé courant
            if np.random.rand() < epsilon:
                a_k = np.random.randint(0, act_dim)
            else:
                s_t = torch.FloatTensor(s_sim).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    a_k = int(qnet(s_t).argmax().item())

            future_acts.append(a_k)

            # Simuler le prochain état avec WM_C3 (1 step fiable)
            # sauf pour le dernier step (on n'a pas besoin de s_sim après)
            if k < n_step - 1:
                s_sim = wm_c3.predict(s_sim, a_k)

        # ── Étape 2 : prédiction directe à n steps ───────────
        s_norm = self._norm(state)
        a_vec  = np.concatenate([self._onehot(a) for a in future_acts])

        s_t = torch.FloatTensor(s_norm).unsqueeze(0).to(self.device)
        a_t = torch.FloatTensor(a_vec).unsqueeze(0).to(self.device)

        with torch.no_grad():
            s_tn_norm = self.model.predict(s_t, a_t).cpu().numpy()[0]

        s_tn = self._clip(self._denorm(s_tn_norm))
        return s_tn, future_acts

    # ── Règles physiques par environnement ───────────────────

    def is_done(self, s_next: np.ndarray) -> bool:
        """Règle physique de terminaison (pas de tête neuronale dédiée)."""
        env = self.env_name
        if env == "CartPole-v1":
            return bool(abs(s_next[0]) > 2.4 or abs(s_next[2]) > 0.209)
        if env == "MountainCar-v0":
            return bool(s_next[0] >= 0.5)
        if env == "LunarLander-v2":
            return bool(s_next[1] <= 0.0)
        return False

    def synthetic_reward(self, s_next: np.ndarray, done: bool) -> float:
        """Reward synthétique selon les règles de l'environnement."""
        env = self.env_name
        if env == "CartPole-v1":
            return 0.0 if done else 1.0
        if env == "MountainCar-v0":
            return 0.0 if done else -1.0
        if env == "LunarLander-v2":
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
              f"(env={wrapper.env_name}, "
              f"config={model.config}, n_step={model.n_step})")
        return wrapper


# ════════════════════════════════════════════════════════════
#  3. CONSTRUCTION DU DATASET
# ════════════════════════════════════════════════════════════

def build_dataset(data: dict, config: int, n_step: int,
                  device: torch.device, split : str = "train") -> tuple:
    """
    Construit (X, Y) depuis le dict numpy fourni par data_collector.

    data   : dict avec clés states, actions, next_states, dones, mean, std
    config : 1, 2 ou 3
    n_step : utilisé seulement si config == 2

    Config 1 : X = [s_norm | a_oh]            Y = s2_norm
    Config 3 : X = [s_norm | a_oh]            Y = s2_norm - s_norm  (ΔS)
    Config 2 : X = [s0_norm | a0_oh...an_oh]  Y = s_tn_norm
               → séquences consécutives, épisodes non traversés
    """
    mean = data["mean"]
    std  = data["std"] + 1e-8

    if config == 2:
        return _build_multistep(data, n_step, mean, std, device, split)

    # ── Configs 1 & 3 ────────────────────────────────────────
    # split: "train", "val", "test"
    states  = data[f"states_{split}"]
    next_s  = data[f"next_states_{split}"]
    actions = data[f"actions_{split}"]
    act_dim = int(actions.max()) + 1


    s_norm  = (states - mean) / std
    s2_norm = (next_s - mean) / std
    a_oh    = np.eye(act_dim, dtype=np.float32)[actions]  # (N, act_dim)

    X = np.concatenate([s_norm, a_oh], axis=1)           # (N, obs+act)
    Y = s2_norm if config == 1 else (s2_norm - s_norm)   # (N, obs_dim)

    print(f"[DATA] Config {config} — {X.shape[0]:,} samples")
    return (torch.FloatTensor(X).to(device),
            torch.FloatTensor(Y).to(device))


def _build_multistep(data: dict, n_step: int,
                     mean: np.ndarray, std: np.ndarray,
                     device: torch.device, split: str = "train") -> tuple:
    """
    Config 2 — construit des séquences (S0, [A0..An-1]) → S(t+n).

    Travaille directement sur les numpy arrays de data_collector.
    Rejette toute séquence qui traverse une limite d'épisode (done=True).

    Exemple pour n_step=3 :
      i=0 : done[0]=False, done[1]=False → séquence valide
             X = [s_norm[0] | a_oh[0] | a_oh[1] | a_oh[2]]
             Y = s_norm[3]   (= next_states[2] normalisé)

      i=2 : done[2]=True → séquence rejetée (épisode terminé au milieu)
    """
    states      = data[f"states_{split}"]       # (N, obs_dim)
    next_states = data[f"next_states_{split}"]  # (N, obs_dim)
    actions     = data[f"actions_{split}"]      # (N,)
    dones       = data[f"dones_{split}"]        # (N,)  float 0.0/1.0
    N           = len(states)
    act_dim     = int(actions.max()) + 1

    X_list, Y_list = [], []
    rejected = 0

    for i in range(N - n_step):
        # Rejeter si un done apparaît dans les n-1 premiers steps
        # (le dernier step peut être done, c'est S(t+n) qui nous intéresse)
        if np.any(dones[i : i + n_step - 1] > 0.5):
            rejected += 1
            continue

        # État de départ normalisé
        s0_norm = (states[i] - mean) / std

        # État d'arrivée = next_state du (n_step-1)-ième step
        stn_norm = (next_states[i + n_step - 1] - mean) / std

        # Concaténer les n actions en one-hot
        a_seq = np.zeros(n_step * act_dim, dtype=np.float32)
        for k in range(n_step):
            a_seq[k * act_dim + int(actions[i + k])] = 1.0

        X_list.append(np.concatenate([s0_norm, a_seq]))
        Y_list.append(stn_norm)

    total    = N - n_step
    kept     = len(X_list)
    pct_kept = 100 * kept / max(total, 1)

    print(f"[DATA] Config 2 (n={n_step}) — "
          f"{kept:,} séquences valides / {total:,} possibles "
          f"({pct_kept:.1f}% conservées, {rejected:,} rejetées)")

    if kept == 0:
        raise ValueError(
            "Aucune séquence multi-step valide. "
            "Les épisodes sont probablement trop courts pour n_step={n_step}. "
            "Essaie --nstep 2 ou collecte plus de données."
        )

    X = torch.FloatTensor(np.array(X_list)).to(device)
    Y = torch.FloatTensor(np.array(Y_list)).to(device)
    return X, Y


# ════════════════════════════════════════════════════════════
#  4. COLLECTE INTERNE (fallback si pas de data_collector.py)
# ════════════════════════════════════════════════════════════

def _collect_fallback(env_name: str, n_steps: int, seed: int) -> dict:
    """
    Collecte basique utilisée uniquement si data_collector.py est absent.
    Politique 100% aléatoire — préférer data_collector.py pour MountainCar.
    """
    print(f"[WARN] data_collector.py non trouvé — collecte interne ({n_steps} steps)")
    env  = gym.make(env_name)
    env.reset(seed=seed)
    obs, _ = env.reset()
    N      = n_steps

    states      = np.zeros((N, env.observation_space.shape[0]), dtype=np.float32)
    actions     = np.zeros(N, dtype=np.int32)
    rewards     = np.zeros(N, dtype=np.float32)
    next_states = np.zeros((N, env.observation_space.shape[0]), dtype=np.float32)
    dones       = np.zeros(N, dtype=np.float32)

    for i in range(N):
        a = env.action_space.sample()
        ns, r, term, trunc, _ = env.step(a)
        done = term or trunc
        states[i]      = obs
        actions[i]     = a
        rewards[i]     = r
        next_states[i] = ns
        dones[i]       = float(done)
        obs = ns if not done else env.reset()[0]

    env.close()
    mean = states.mean(axis=0).astype(np.float32)
    std  = states.std(axis=0).astype(np.float32)
    return dict(states=states, actions=actions, rewards=rewards,
                next_states=next_states, dones=dones,
                mean=mean, std=std, env_name=env_name)

class EarlyStopping:
    def __init__(self, patience=8, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float('inf')
        self.num_bad = 0
    def step(self, loss):
        if loss < self.best - self.min_delta:
            self.best = loss
            self.num_bad = 0
            return False
        else:
            self.num_bad += 1
            return self.num_bad >= self.patience


# ════════════════════════════════════════════════════════════
#  5. ENTRAÎNEMENT
# ════════════════════════════════════════════════════════════

def train_world_model(env_name: str, config: int = 3,
                      n_step: int = 3, hidden: int = HIDDEN,
                      epochs: int = EPOCHS, seed: int = SEED,
                      collect_steps: int = COLLECT_STEPS,
                      data_path: str = None) -> WorldModelWrapper:
    """
    Pipeline complet :
      1. Charger les données depuis data_path (recommandé)
         ou collecter automatiquement (fallback)
      2. Construire le dataset selon la config
      3. Entraîner le WorldModel
      4. Sauvegarder le WorldModelWrapper

    Retourne un WorldModelWrapper prêt à l'emploi.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  WorldModel | env={env_name} | config={config}"
          + (f" | n_step={n_step}" if config == 2 else ""))
    print(f"  device={device} | hidden={hidden} | epochs={epochs}")
    print(f"{'='*60}\n")

    # ── Dimensions depuis l'env ──────────────────────────────
    env     = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    env.close()

    # ── Chargement des données ───────────────────────────────
    if data_path and os.path.exists(data_path):
        print(f"[DATA] Chargement depuis {data_path} via numpy")
        try:
            from collect_data.data_collect_general import load_dataset
            print(f"[DATA] Chargement depuis {data_path} via data_collect_general.load_dataset()")
            data = load_dataset(data_path)
        except ImportError:
            npz  = np.load(data_path, allow_pickle=True)
            data = {k: npz[k] for k in npz.files}
            data["env_name"] = str(data.get("env_name", env_name))
    else:
        print(f"[WARN] {data_path} introuvable ou non spécifié — collecte automatique.")
        if data_path:
            print(f"[WARN] {data_path} introuvable — collecte automatique.")
        data = _collect_fallback(env_name, collect_steps, seed)

    print(f"[NORM] mean = {np.round(data['mean'], 4)}")
    print(f"[NORM] std  = {np.round(data['std'],  4)}\n")

    # ── Dataset ──────────────────────────────────────────────
    X_train, Y_train = build_dataset(data, config, n_step, device, split="train")
    X_val,   Y_val   = build_dataset(data, config, n_step, device, split="val")
    X_test,  Y_test  = build_dataset(data, config, n_step, device, split="test")

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, Y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_test, Y_test), batch_size=BATCH_SIZE, shuffle=False)

    # ── Modèle ───────────────────────────────────────────────
    model   = WorldModel(obs_dim, act_dim, config, n_step, hidden).to(device)
    opt     = optim.Adam(model.parameters(), lr=LR)
    sched   = optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.5)
    loss_fn = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[WM] Paramètres : {n_params:,}")
    print(f"[WM] Samples train={len(X_train):,} | val={len(X_val):,} | test={len(X_test):,}\n")


    # ── Boucle d'entraînement ────────────────────────────────
    best_loss, best_sd = np.inf, None
    tr_losses, val_losses = [], []
    early_stopper = EarlyStopping()
    best_val = float("inf")              # CORRECT pour le tracking val_loss


    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            # Séparer state et action_input depuis X
            # X = [s_norm (obs_dim) | a_input (act_dim ou n*act_dim)]
            xb, yb = xb.to(device), yb.to(device)
            s_b = xb[:, :obs_dim]
            a_b = xb[:, obs_dim:]         # act_dim pour C1/C3, n*act_dim pour C2

            pred = model.forward(s_b, a_b)
            loss = nn.MSELoss()(pred, yb)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()*len(xb)
        train_loss /= len(train_loader.dataset)
        tr_losses.append(train_loss)


        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                s_b = xb[:, :obs_dim]
                a_b = xb[:, obs_dim:]
                pred = model.forward(s_b, a_b)
                val_loss += nn.MSELoss()(pred, yb).item() * len(xb)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        # Scheduler
        sched.step()


        # EarlyStopping & checkpointing
        if val_loss < best_val:
            best_val = val_loss
            best_sd  = {k: v.clone() for k, v in model.state_dict().items()}

        if early_stopper.step(val_loss):
            print(f"Early stopping à {epoch}")
            break

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}, train={train_loss:.6f}, val={val_loss:.6f}, lr={sched.get_last_lr()[0]:.1e}")

        # if avg < best_loss:
        #     best_loss = avg
        #     best_sd   = {k: v.clone() for k, v in model.state_dict().items()}

        # if epoch % 5 == 0 or epoch == 1:
        #     print(f"  Epoch {epoch:3d}/{epochs} | "
        #           f"loss={avg:.6f} | best={best_loss:.6f} | "
        #           f"lr={sched.get_last_lr()[0]:.1e}")

    # ── Restaurer meilleur checkpoint ────────────────────────
    model.load_state_dict(best_sd)
    model.eval()

    # ── Sauvegarde ───────────────────────────────────────────
    tag      = f"config{config}" + (f"_n{n_step}" if config == 2 else "")
    savepath = str(CKPT_DIR / f"wm_{env_name}_{tag}.pth")
    wrapper  = WorldModelWrapper(model, data["mean"], data["std"],
                                  env_name, device)
    wrapper.save(savepath)

    # ── Évaluation rapide (configs 1 & 3 uniquement) ─────────

    if config != 2:
        _quick_eval(wrapper, data, split="test")
        evaluate_on_test(model, X_test, Y_test, device)

    return wrapper



def evaluate_on_test(model, X_test, Y_test, device):
    model.eval()
    preds = []
    targets = []
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=256, shuffle=False)
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            s_b = xb[:, :model.obs_dim]
            a_b = xb[:, model.obs_dim:]
            p = model.forward(s_b, a_b).cpu().numpy()
            preds.append(p)
            targets.append(yb.cpu().numpy())
    pred = np.concatenate(preds)
    y_true = np.concatenate(targets)
    mse = np.mean((pred - y_true)**2)
    mae = np.mean(np.abs(pred - y_true))
    print(f"\n[TEST] MSE={mse:.6f} | MAE={mae:.6f}\n")
    return mse, mae


# def _quick_eval(wrapper: WorldModelWrapper, data: dict, n_eval: int = 500):
#     """MAE sur n_eval transitions pour configs 1 & 3."""
#     idx = np.random.choice(len(data["states"]), size=n_eval, replace=False)
#     errors = [
#         np.abs(wrapper.predict(data["states"][i], int(data["actions"][i]))
#                - data["next_states"][i]).mean()
#         for i in idx
#     ]
#     print(f"\n[EVAL] MAE sur {n_eval} transitions : "
#           f"{np.mean(errors):.5f} ± {np.std(errors):.5f}  "
#           f"(max={np.max(errors):.5f})\n")
    
def _quick_eval(wrapper: WorldModelWrapper, data: dict, n_eval: int = 500, split="test"):
    """MAE sur n_eval transitions (par défaut dans le test set)."""
    N = len(data[f"states_{split}"])
    n_eval = min(n_eval, N)  # au cas où
    idx = np.random.choice(N, size=n_eval, replace=False)
    errors = [
        np.abs(
            wrapper.predict(data[f"states_{split}"][i], int(data[f"actions_{split}"][i]))
            - data[f"next_states_{split}"][i]
        ).mean() for i in idx
    ]
    print(f"\n[EVAL] MAE sur {n_eval} transitions ({split} set) : "
          f"{np.mean(errors):.5f} ± {np.std(errors):.5f}  "
          f"(max={np.max(errors):.5f})\n")


# ════════════════════════════════════════════════════════════
#  6. MAIN
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Entraîne un WorldModel (config 1/2/3) sur un env Gymnasium")
    parser.add_argument("--env",    default="CartPole-v1",
                        choices=["CartPole-v1", "MountainCar-v0",
                                 "LunarLander-v2", "Acrobot-v1"])
    parser.add_argument("--config", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--nstep",  type=int, default=3,
                        help="Horizon n pour config 2 (défaut : 3)")
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed",   type=int, default=SEED)
    parser.add_argument("--data",   default=None,
                        help="Chemin vers .npz de data_collector.py "
                             "(ex: data/data_CartPole-v1.npz). "
                             "Si absent → collecte automatique.")
    parser.add_argument("--collect", type=int, default=COLLECT_STEPS,
                        help="Steps collectés si --data absent (défaut: 30000)")
    parser.add_argument("--all",    action="store_true",
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