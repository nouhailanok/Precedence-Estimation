"""
train_dnn_advanced.py — Precedence Estimation sur CartPole-v1
World model : f(s_t, a_t) → Δs_t   puis   s_{t+1} = s_t + Δs_t

Pourquoi prédire Δs plutôt que s' ?
  - Δs est centré autour de 0 → plus facile à optimiser
  - Exploite la continuité physique de CartPole
  - Moins d'accumulation d'erreur dans le rollout séquentiel

Choix de loss : MSE ou Huber (configurable via LOSS_TYPE)

Fonctionnalités :
  - Mixed policy dataset (cartpole_data_mixed_policy.npz)
  - Predict delta Δs
  - MSE ou HuberLoss
  - Validation split + early stopping
  - LR scheduler (ReduceLROnPlateau)
  - Checkpoint best model
  - AdamW optimizer
  - Évaluation physique (dénormalisée)
  - Rollout séquentiel (multiple seeds, stats d'erreur)
  - Visualisations

Usage :
    python train_dnn_advanced.py
"""

import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import gymnasium as gym
import os


# ════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════

DATA_PATH       = "cartpole_data_mixed_policy.npz"
CHECKPOINT_PATH = "best_transition_dnn.pth"
PLOTS_DIR = "winner_model_plots"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE  = 256
EPOCHS      = 200
LR          = 1e-3
PATIENCE    = 15
MIN_DELTA   = 1e-6

# ==== LOSS CHOICE ====
# "mse"   → nn.MSELoss()        — standard, adapté à CartPole (recommandé)
# "huber" → nn.HuberLoss(delta) — robuste aux outliers (mieux sur LunarLander)
LOSS_TYPE   = "mse"
HUBER_DELTA = 1.0

# ==== ROLLOUT CONFIG ====
N_ROLLOUT_EPISODES = 5     # nombre d'épisodes pour évaluer le rollout
ROLLOUT_STEPS      = 60    # pas max par épisode de rollout

SEED = 42


# ════════════════════════════════════════════════════════════
# REPRODUCTIBILITÉ
# ════════════════════════════════════════════════════════════

def set_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

set_seed()
os.makedirs(PLOTS_DIR, exist_ok=True)
print(f"Device : {DEVICE}  |  Loss : {LOSS_TYPE.upper()}"
      + (f"  (delta={HUBER_DELTA})" if LOSS_TYPE == "huber" else ""))


# ════════════════════════════════════════════════════════════
# LOSS FACTORY
# ════════════════════════════════════════════════════════════

def get_loss_fn(loss_type: str = LOSS_TYPE, delta: float = HUBER_DELTA) -> nn.Module:
    """
    Retourne la fonction de loss choisie.

    MSE   → pénalise quadratiquement toutes les erreurs.
            Idéal quand les résidus Δs sont petits et gaussiens (CartPole).
            Sensible aux outliers (les grandes erreurs dominent).

    Huber → MSE si |erreur| < delta, MAE sinon.
            Robuste aux transitions aberrantes.
            Utile si la politique mixte génère des transitions
            inhabituelles (chutes brusques, états limites).
            Recommandé pour LunarLander ou des envs bruités.

    Conseil : commencer avec MSE, switcher sur Huber si la loss
    d'entraînement oscille beaucoup ou si MSE_test >> MSE_train.
    """
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "huber":
        return nn.HuberLoss(delta=delta)
    else:
        raise ValueError(f"loss_type doit être 'mse' ou 'huber'. Reçu : '{loss_type}'")


# ════════════════════════════════════════════════════════════
# CHARGEMENT + PRÉPARATION DES DONNÉES
# ════════════════════════════════════════════════════════════

def load_data(path: str):
    """
    Charge le NPZ et construit les cibles Δs = s_{t+1} - s_t.

    Contenu attendu du NPZ (produit par collect_data_dqn_v2) :
        s_train / s_val / s_test    (N, 4) normalisés
        a_train / a_val / a_test    (N, 2) one-hot
        sn_train / sn_val / sn_test (N, 4) normalisés
        mean (1, 4), std (1, 4)

    Pourquoi Δs normalisé ?
        Δs = sn_norm - s_norm  (les deux sont déjà dans l'espace normalisé)
        → Δs est centré en 0, amplitude ~3× plus petite que s'
        → le réseau apprend une correction résiduelle, pas une valeur absolue
        → reconstruction : s'_norm = s_norm + Δs_prédit
                           s'_phys = denormalize(s'_norm)
    """
    data = np.load(path)

    def t(k): return torch.tensor(data[k], dtype=torch.float32)

    s_tr,  a_tr,  sn_tr  = t("s_train"), t("a_train"), t("sn_train")
    s_val, a_val, sn_val = t("s_val"),   t("a_val"),   t("sn_val")
    s_te,  a_te,  sn_te  = t("s_test"),  t("a_test"),  t("sn_test")

    mean = data["mean"].astype(np.float32)   # (1, 4)
    std  = data["std"].astype(np.float32)    # (1, 4)

    # Protection contre std ≈ 0 (dimensions constantes)
    std = np.where(std < 1e-8, 1.0, std)

    # ── Cibles : Δs = s_{t+1} - s_t  (dans l'espace normalisé) ──
    delta_tr  = sn_tr  - s_tr
    delta_val = sn_val - s_val
    delta_te  = sn_te  - s_te

    print(f"\nTrain      : {len(s_tr):,} transitions")
    print(f"Validation : {len(s_val):,} transitions")
    print(f"Test       : {len(s_te):,} transitions")
    print(f"Entrée DNN : {s_tr.shape[1]}D état + {a_tr.shape[1]}D action = 6D")
    print(f"Cible DNN  : Δs ({delta_tr.shape[1]}D) — prédit delta, pas s' directement")
    print(f"  |Δs| moyen (train) : {delta_tr.abs().mean():.5f}  "
          f"vs  |s'| moyen : {sn_tr.abs().mean():.5f}")

    return (s_tr, a_tr, delta_tr,
            s_val, a_val, delta_val,
            s_te,  a_te,  delta_te,
            sn_te,   # ← état suivant réel (pour évaluation physique)
            mean, std)


# ════════════════════════════════════════════════════════════
# DÉNORMALISATION
# ════════════════════════════════════════════════════════════

def denormalize(s_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Repasse de l'espace normalisé aux unités physiques réelles."""
    return s_norm * std + mean


# ════════════════════════════════════════════════════════════
# ARCHITECTURE DNN
# ════════════════════════════════════════════════════════════

class TransitionDNN(nn.Module):
    """
    World model — prédit Δs = s_{t+1} - s_t.

    Entrée  : concat[s_t normalisé (4D), a_t one-hot (2D)] = 6D
    Sortie  : Δs normalisé prédit (4D)
    Reconstruction : s_{t+1} = s_t + Δs_prédit  (dans l'espace normalisé)

    Architecture Tiny [32-32] — meilleure sur CartPole (MSE=5e-7, R²=1.0)
    validée sur 5 seeds vs 13 autres architectures via dnn_compare_robust.py
    """
    def __init__(self, state_dim: int = 4, action_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, state_dim),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Retourne Δs prédit (pas s' directement)."""
        return self.net(torch.cat([state, action], dim=1))

    def predict_next(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Reconstruit s_{t+1} = s_t + Δs prédit (dans l'espace normalisé)."""
        return state + self.forward(state, action)


# ════════════════════════════════════════════════════════════
# EARLY STOPPING
# ════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience: int = PATIENCE, min_delta: float = MIN_DELTA):
        self.patience  = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter   = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ════════════════════════════════════════════════════════════
# ENTRAÎNEMENT
# ════════════════════════════════════════════════════════════

def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion):
    """
    Boucle d'entraînement avec :
      - MSE ou HuberLoss sur Δs (pas sur s')
      - Validation à chaque époque
      - Checkpoint du meilleur modèle (val loss)
      - Early stopping
      - LR scheduler ReduceLROnPlateau

    Note sur la loss :
      Minimiser sur Δs est équivalent à minimiser sur s' car s_t est fixé,
      mais Δs a une amplitude plus petite → gradients plus stables.
    """
    stopper          = EarlyStopping()
    best_val         = float("inf")
    tr_losses, val_losses = [], []
    t0 = time.time()

    for epoch in range(EPOCHS):

        # ── Train ──
        model.train()
        tr_loss = 0.0
        for s, a, delta in train_loader:
            s, a, delta = s.to(DEVICE), a.to(DEVICE), delta.to(DEVICE)
            pred = model(s, a)
            loss = criterion(pred, delta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * s.size(0)
        tr_loss /= len(train_loader.dataset)

        # ── Validation ──
        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for s, a, delta in val_loader:
                s, a, delta = s.to(DEVICE), a.to(DEVICE), delta.to(DEVICE)
                vl_loss += criterion(model(s, a), delta).item() * s.size(0)
        vl_loss /= len(val_loader.dataset)

        scheduler.step(vl_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        tr_losses.append(tr_loss)
        val_losses.append(vl_loss)

        print(f"Epoch {epoch+1:03d}/{EPOCHS}  "
              f"train={tr_loss:.7f}  val={vl_loss:.7f}  lr={current_lr:.6f}")

        if vl_loss < best_val:
            best_val = vl_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print("  ✓ checkpoint sauvegardé")

        if stopper.step(vl_loss):
            print(f"\nEarly stopping à l'époque {epoch+1}  "
                  f"(best_val={best_val:.7f})")
            break

    print(f"\nEntraînement terminé en {time.time()-t0:.1f}s  —  "
          f"best val {LOSS_TYPE.upper()} = {best_val:.7f}")
    return tr_losses, val_losses


# ════════════════════════════════════════════════════════════
# ÉVALUATION
# ════════════════════════════════════════════════════════════

def evaluate(model, test_loader, s_te_tensor, sn_te_tensor, mean, std, eps=0.01):
    """
    Deux niveaux d'évaluation :

    1. Sur Δs normalisé → cohérence avec la loss d'entraînement
    2. Sur s' physique (après reconstruction + dénormalisation)
       → interprétable en unités réelles (m, m/s, rad, rad/s)

    Reconstruction :
        Δs_pred_norm = model(s_norm, a)
        s'_norm      = s_norm + Δs_pred_norm
        s'_phys      = denormalize(s'_norm)
    """
    criterion_eval = get_loss_fn(LOSS_TYPE, HUBER_DELTA)
    model.eval()

    deltas_pred, deltas_real = [], []
    with torch.no_grad():
        for s, a, delta in test_loader:
            s, a = s.to(DEVICE), a.to(DEVICE)
            deltas_pred.append(model(s, a).cpu().numpy())
            deltas_real.append(delta.numpy())

    dp = np.concatenate(deltas_pred)
    dr = np.concatenate(deltas_real)

    # Loss sur Δs (MSE toujours pour comparaison équitable)
    # loss_delta = float(np.mean((dp - dr) ** 2))
    loss_delta = float(np.mean(
        (dp - dr)**2 if LOSS_TYPE == "mse"
        else np.where(np.abs(dp-dr) < HUBER_DELTA,
                      0.5*(dp-dr)**2,
                      HUBER_DELTA*(np.abs(dp-dr) - 0.5*HUBER_DELTA))
    ))

    # Reconstruction s' et métriques physiques
    s_norm      = s_te_tensor.numpy()
    sn_norm     = sn_te_tensor.numpy()
    spred_norm  = s_norm + dp
    spred_phys  = denormalize(spred_norm, mean, std)
    sreal_phys  = denormalize(sn_norm,    mean, std)

    err_phys     = spred_phys - sreal_phys
    mse_per_dim  = np.mean(err_phys ** 2,      axis=0)
    mae_per_dim  = np.mean(np.abs(err_phys),   axis=0)
    rmse_per_dim = np.sqrt(mse_per_dim)

    ss_res = np.sum((sreal_phys - spred_phys) ** 2)
    ss_tot = np.sum((sreal_phys - sreal_phys.mean(0)) ** 2)
    r2     = 1 - ss_res / (ss_tot + 1e-10)

    within  = np.abs(spred_norm - sn_norm) < eps
    acc_all = within.all(axis=1).mean()

    dim_names = ["x (m)", "ẋ (m/s)", "θ (rad)", "θ̇ (rad/s)"]
    print(f"\n{'─'*62}")
    print(f"MSE sur Δs  (test, normalisé) : {loss_delta:.7f}")
    print(f"R²   sur s' (physique)        : {r2:.6f}")
    print(f"Accuracy ε={eps} (normalisé)  : {acc_all:.4f}")
    print(f"\n{'Dim':12s} {'MSE':>12} {'RMSE':>12} {'MAE':>12}")
    print(f"{'─'*50}")
    for i, name in enumerate(dim_names):
        print(f"{name:12s} {mse_per_dim[i]:>12.8f} "
              f"{rmse_per_dim[i]:>12.6f} {mae_per_dim[i]:>12.6f}")
    print(f"{'─'*62}")

    model.train()
    return spred_phys, sreal_phys, {
        "loss_delta":  loss_delta,
        "r2":          r2,
        "acc":         acc_all,
        "mse_per_dim": mse_per_dim,
        "rmse_per_dim":rmse_per_dim,
        "mae_per_dim": mae_per_dim,
    }


# ════════════════════════════════════════════════════════════
# ROLLOUT — FONCTIONS UTILITAIRES
# ════════════════════════════════════════════════════════════

def collect_rollout_episode(env, model, mean, std,
                            max_steps: int = ROLLOUT_STEPS,
                            seed: int = 0) -> dict:
    """
    Collecte UN épisode de rollout complet.

    À chaque pas :
        1. Predict  : Δs_pred = model(s_norm, a_oh)
                      s'_norm = s_norm + Δs_pred        ← AVANT env.step
        2. Execute  : s'_real = env.step(a)              ← vrai environnement
        3. Compare  : erreur physique pas à pas

    Retourne un dict avec les trajectoires et erreurs.
    C'est la démonstration centrale de la Precedence Estimation :
    le world model prédit s_{t+1} AVANT d'appeler env.step().
    """
    obs, _ = env.reset(seed=seed)
    model.eval()

    predicted_phys  = []
    real_phys       = []
    actions_taken   = []
    errors_per_step = []
    step_count      = 0

    with torch.no_grad():
        for _ in range(max_steps):
            action = env.action_space.sample()

            # ── PRÉCÉDENCE : prédire AVANT env.step ──
            s_norm = (obs - mean.squeeze()) / std.squeeze()
            s_t    = torch.tensor(s_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            a_oh   = torch.zeros(1, 2, device=DEVICE)
            a_oh[0, action] = 1.0

            spred_norm = model.predict_next(s_t, a_oh).cpu().numpy().squeeze()
            spred_phys_step = denormalize(spred_norm[np.newaxis], mean, std).squeeze()

            # ── Environnement réel ──
            next_obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            err = np.abs(spred_phys_step - next_obs)
            predicted_phys.append(spred_phys_step)
            real_phys.append(next_obs.copy())
            actions_taken.append(action)
            errors_per_step.append(err)
            step_count += 1

            if done:
                break
            obs = next_obs

    return {
        "predicted":  np.array(predicted_phys),   # (T, 4)
        "real":       np.array(real_phys),         # (T, 4)
        "actions":    np.array(actions_taken),     # (T,)
        "errors":     np.array(errors_per_step),   # (T, 4)
        "n_steps":    step_count,
    }


def run_rollout_evaluation(model, mean, std,
                           n_episodes: int = N_ROLLOUT_EPISODES) -> dict:
    """
    Évalue le rollout sur N épisodes et agrège les stats.

    Métriques agrégées :
      - MAE moyen par dimension sur tous les épisodes
      - Erreur max (worst case)
      - Drift : comment l'erreur évolue au fil des pas (s'accumule-t-elle ?)

    Retourne aussi les données brutes du meilleur et pire épisode
    pour les visualisations.
    """
    env = gym.make("CartPole-v1")
    episodes_data = []

    print(f"\nRollout : évaluation sur {n_episodes} épisodes...")
    for ep in range(n_episodes):
        data = collect_rollout_episode(env, model, mean, std, seed=ep * 10 + 77)
        mae  = data["errors"].mean()
        episodes_data.append(data)
        print(f"  Épisode {ep+1} : {data['n_steps']:>3} pas  |  MAE = {mae:.6f}")

    env.close()

    # Agréger toutes les erreurs
    all_errors = np.concatenate([d["errors"] for d in episodes_data], axis=0)
    mae_per_dim = all_errors.mean(axis=0)
    max_err_per_dim = all_errors.max(axis=0)

    # Drift : erreur moyenne par position dans l'épisode (pad avec nan)
    max_len = max(d["n_steps"] for d in episodes_data)
    drift_matrix = np.full((n_episodes, max_len, 4), np.nan)
    for i, d in enumerate(episodes_data):
        T = d["n_steps"]
        drift_matrix[i, :T, :] = d["errors"]
    drift_mean = np.nanmean(drift_matrix, axis=0)   # (max_len, 4)

    dim_names = ["x (m)", "ẋ (m/s)", "θ (rad)", "θ̇ (rad/s)"]
    print(f"\n{'─'*55}")
    print(f"Rollout — agrégé sur {n_episodes} épisodes :")
    print(f"{'Dim':12s} {'MAE moy.':>12} {'Erreur max':>12}")
    for i, name in enumerate(dim_names):
        print(f"{name:12s} {mae_per_dim[i]:>12.6f} {max_err_per_dim[i]:>12.6f}")
    print(f"{'─'*55}")

    # Meilleur et pire épisode selon MAE globale
    maes      = [d["errors"].mean() for d in episodes_data]
    best_ep   = episodes_data[int(np.argmin(maes))]
    worst_ep  = episodes_data[int(np.argmax(maes))]

    return {
        "episodes":       episodes_data,
        "mae_per_dim":    mae_per_dim,
        "max_err_per_dim":max_err_per_dim,
        "drift_mean":     drift_mean,
        "best_ep":        best_ep,
        "worst_ep":       worst_ep,
    }


# ════════════════════════════════════════════════════════════
# VISUALISATIONS
# ════════════════════════════════════════════════════════════

def plot_training_curve(tr_losses, val_losses):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(tr_losses,  label="Train",      color="#378ADD", linewidth=1.5)
    ax.plot(val_losses, label="Validation", color="#D85A30", linewidth=1.5)
    ax.set_xlabel("Époque")
    ax.set_ylabel(f"Loss ({LOSS_TYPE.upper()})")
    ax.set_title(f"Courbe d'entraînement — World Model Δs  [{LOSS_TYPE.upper()}]")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "training_curve.png"), dpi=130)
    plt.show()
    print("Sauvegardé → training_curve.png")


def plot_predictions(pred_phys, real_phys, n=400):
    """Scatter s' prédit vs réel pour chaque dimension."""
    dim_names = ["x (m)", "ẋ (m/s)", "θ (rad)", "θ̇ (rad/s)"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for i, (ax, name) in enumerate(zip(axes, dim_names)):
        r, p = real_phys[:n, i], pred_phys[:n, i]
        ax.scatter(r, p, alpha=0.3, s=8, color="#378ADD")
        lo, hi = min(r.min(), p.min()), max(r.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2)
        r2 = 1 - np.sum((r-p)**2) / (np.sum((r-r.mean())**2) + 1e-10)
        ax.set_title(f"{name}  R²={r2:.4f}", fontsize=10)
        ax.set_xlabel("Réel")
        ax.set_ylabel("Prédit")
        ax.grid(alpha=0.3)
    plt.suptitle(f"Prédiction s' (reconstruit via Δs) vs Réalité — [{LOSS_TYPE.upper()}]",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "predictions_scatter.png"), dpi=130)
    plt.show()
    print("Sauvegardé → predictions_scatter.png")


def plot_delta_distribution(delta_pred_norm, delta_real_norm):
    """Histogrammes superposés Δs prédit vs réel par dimension."""
    dim_names = ["Δx", "Δẋ", "Δθ", "Δθ̇"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    for i, (ax, name) in enumerate(zip(axes, dim_names)):
        ax.hist(delta_real_norm[:, i], bins=50, alpha=0.5,
                color="#1D9E75", label="Δs réel",   density=True)
        ax.hist(delta_pred_norm[:, i], bins=50, alpha=0.5,
                color="#D85A30", label="Δs prédit", density=True)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("valeur normalisée")
        ax.legend(fontsize=8)
    plt.suptitle("Distribution des deltas : prédit vs réel — espace normalisé",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "delta_distribution.png"), dpi=130)
    plt.show()
    print("Sauvegardé → delta_distribution.png")


def plot_rollout_episode(ep_data: dict, title_suffix: str = ""):
    """
    Figure 1 du rollout : trajectoires prédite vs réelle sur 4 dimensions
    pour UN épisode. Montre la qualité pas à pas de la Precedence Estimation.
    """
    pred    = ep_data["predicted"]
    real    = ep_data["real"]
    errors  = ep_data["errors"]
    T       = ep_data["n_steps"]

    dim_names = ["x (m)", "ẋ (m/s)", "θ (rad)", "θ̇ (rad/s)"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    axes = axes.flatten()

    for i, (ax, name) in enumerate(zip(axes, dim_names)):
        ax.plot(range(T), real[:, i],
                color="#1D9E75", linewidth=2, label="Réel")
        ax.plot(range(T), pred[:, i],
                color="#D85A30", linewidth=1.5, linestyle="--",
                label="Prédit (via Δs)")
        ax.fill_between(range(T),
                        real[:, i] - errors[:, i],
                        real[:, i] + errors[:, i],
                        alpha=0.15, color="#D85A30", label="±erreur abs.")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Pas de temps")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    mae = errors.mean()
    plt.suptitle(
        f"Rollout Precedence Estimation — Predict Δs  [{LOSS_TYPE.upper()}]\n"
        f"{title_suffix}  |  {T} pas  |  MAE = {mae:.6f}",
        fontsize=11
    )
    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, f"rollout_{title_suffix.lower().replace(' ', '_')}.png")
    plt.savefig(fname, dpi=130)
    plt.show()
    print(f"Sauvegardé → {fname}")


def plot_rollout_drift(drift_mean: np.ndarray):
    """
    Figure 2 du rollout : drift de l'erreur au fil des pas.
    Montre si l'erreur s'accumule (problème) ou reste stable (bon world model).
    C'est la figure la plus critique pour évaluer la qualité du world model
    dans un contexte model-based RL.
    """
    T         = drift_mean.shape[0]
    dim_names = ["x (m)", "ẋ (m/s)", "θ (rad)", "θ̇ (rad/s)"]
    colors    = ["#378ADD", "#1D9E75", "#D85A30", "#9B59B6"]

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (name, color) in enumerate(zip(dim_names, colors)):
        valid = ~np.isnan(drift_mean[:, i])
        ax.plot(np.where(valid)[0], drift_mean[valid, i],
                label=name, color=color, linewidth=1.8)

    ax.set_xlabel("Pas de temps dans l'épisode")
    ax.set_ylabel("Erreur absolue moyenne (unités physiques)")
    ax.set_title(
        f"Drift de l'erreur de prédiction — Predict Δs  [{LOSS_TYPE.upper()}]\n"
        f"Stable = bon world model  |  Croissant = accumulation d'erreur"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "rollout_drift.png"), dpi=130)
    plt.show()
    print("Sauvegardé → rollout_drift.png")


def plot_rollout_error_boxplot(episodes_data: list):
    """
    Figure 3 du rollout : boxplot de la distribution des erreurs
    sur tous les épisodes, par dimension.
    Montre la variabilité et les outliers.
    """
    all_errors  = np.concatenate([d["errors"] for d in episodes_data], axis=0)
    dim_names   = ["x (m)", "ẋ (m/s)", "θ (rad)", "θ̇ (rad/s)"]

    fig, ax = plt.subplots(figsize=(9, 4))
    bp = ax.boxplot(
        [all_errors[:, i] for i in range(4)],
        labels=dim_names,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )
    colors = ["#378ADD", "#1D9E75", "#D85A30", "#9B59B6"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Erreur absolue (unités physiques)")
    ax.set_title(
        f"Distribution des erreurs de prédiction par dimension\n"
        f"[{LOSS_TYPE.upper()}]  —  {len(episodes_data)} épisodes"
    )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "rollout_error_boxplot.png"), dpi=130)
    plt.show()
    print("Sauvegardé → " + os.path.join(PLOTS_DIR, "rollout_error_boxplot.png"))


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # 1. Chargement
    (s_tr, a_tr, delta_tr,
     s_val, a_val, delta_val,
     s_te,  a_te,  delta_te,
     sn_te, mean, std) = load_data(DATA_PATH)

    # 2. DataLoaders
    train_loader = DataLoader(TensorDataset(s_tr,  a_tr,  delta_tr),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(s_val, a_val, delta_val),
                              batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(TensorDataset(s_te,  a_te,  delta_te),
                              batch_size=BATCH_SIZE, shuffle=False)

    # 3. Modèle
    model    = TransitionDNN().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nArchitecture :\n{model}\nParamètres : {n_params:,}\n")

    # 4. Optimiseur + loss + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = get_loss_fn(LOSS_TYPE, HUBER_DELTA)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    print(f"Loss utilisée : {criterion}")

    # 5. Entraînement
    tr_losses, val_losses = train_model(
        model, train_loader, val_loader, optimizer, scheduler, criterion
    )

    # 6. Charger le meilleur checkpoint
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    print("\nMeilleur checkpoint rechargé.")

    # 7. Courbe d'entraînement
    plot_training_curve(tr_losses, val_losses)

    # 8. Évaluation sur le test set
    pred_phys, real_phys, metrics = evaluate(
        model, test_loader, s_te, sn_te, mean, std
    )
    plot_predictions(pred_phys, real_phys)

    # 9. Distribution des deltas
    model.eval()
    dp_list = []
    with torch.no_grad():
        for s, a, _ in test_loader:
            dp_list.append(model(s.to(DEVICE), a.to(DEVICE)).cpu().numpy())
    model.train()
    delta_pred_norm = np.concatenate(dp_list)
    delta_real_norm = delta_te.numpy()
    plot_delta_distribution(delta_pred_norm, delta_real_norm)

    # ════════════════════════════════════════════════
    # 10. ROLLOUT — évaluation complète sur N épisodes
    # ════════════════════════════════════════════════
    rollout_results = run_rollout_evaluation(model, mean, std,
                                             n_episodes=N_ROLLOUT_EPISODES)

    # Figure A : meilleur épisode
    plot_rollout_episode(rollout_results["best_ep"],
                         title_suffix="Meilleur épisode")

    # Figure B : pire épisode (montre les limites du modèle)
    plot_rollout_episode(rollout_results["worst_ep"],
                         title_suffix="Pire épisode")

    # Figure C : drift de l'erreur au fil des pas
    plot_rollout_drift(rollout_results["drift_mean"])

    # Figure D : boxplot de la distribution des erreurs
    plot_rollout_error_boxplot(rollout_results["episodes"])

    # 11. Sauvegarde finale
    torch.save(model.state_dict(), "transition_dnn_final.pth")
    print("\nModèle sauvegardé → transition_dnn_final.pth")

    print(f"\n{'═'*50}")
    print(f"Résumé final :")
    print(f"  Loss type     : {LOSS_TYPE.upper()}")
    print(f"  R²            : {metrics['r2']:.6f}")
    print(f"  Accuracy ε    : {metrics['acc']:.4f}")
    print(f"  RMSE physique : {metrics['rmse_per_dim'].mean():.6f}")
    print(f"  MAE rollout   : {rollout_results['mae_per_dim'].mean():.6f}")
    print(f"{'═'*50}")
    print("\nFigures générées :")
    print("  training_curve.png              — courbe loss train/val")
    print("  predictions_scatter.png         — scatter s' prédit vs réel")
    print("  delta_distribution.png          — distribution Δs prédit vs réel")
    print("  rollout_meilleur_épisode.png    — trajectoire meilleur épisode")
    print("  rollout_pire_épisode.png        — trajectoire pire épisode")
    print("  rollout_drift.png               — drift de l'erreur par pas")
    print("  rollout_error_boxplot.png       — distribution erreurs par dimension")