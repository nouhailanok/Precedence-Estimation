
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
  - Rollout séquentiel
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


# ════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════

DATA_PATH       = "cartpole_data_mixed_policy.npz"
CHECKPOINT_PATH = "best_transition_dnn.pth"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE  = 256
EPOCHS      = 200
LR          = 1e-3
PATIENCE    = 15
MIN_DELTA   = 1e-6

# ==== LOSS CHOICE ====
# "mse"   → nn.MSELoss()          — standard, adapté à CartPole
# "huber" → nn.HuberLoss(delta)   — robuste aux outliers (mieux sur LunarLander)
LOSS_TYPE = "mse"
HUBER_DELTA = 1.0

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
        s_train / s_val / s_test   (N, 4) normalisés
        a_train / a_val / a_test   (N, 2) one-hot
        sn_train / sn_val / sn_test (N, 4) normalisés
        mean (1, 4), std (1, 4)

    Pourquoi Δs normalisé ?
        Δs = sn_norm - s_norm  (les deux sont déjà dans l'espace normalisé)
        → Δs est centré en 0, amplitude ~10× plus petite que s'
        → le réseau apprend une correction résiduelle, pas une valeur absolue
        → reconstruction : s'_norm = s_norm + Δs_prédit
                           s'_phys = denormalize(s'_norm)
    """
    data = np.load(path)

    def t(k): return torch.tensor(data[k], dtype=torch.float32)

    s_tr, a_tr, sn_tr = t("s_train"), t("a_train"), t("sn_train")
    s_val, a_val, sn_val = t("s_val"),  t("a_val"),  t("sn_val")
    s_te, a_te, sn_te = t("s_test"),  t("a_test"),  t("sn_test")

    mean = data["mean"].astype(np.float32)   # (1, 4)
    std  = data["std"].astype(np.float32)    # (1, 4)

    # ── Cibles : Δs = s_{t+1} - s_t  (dans l'espace normalisé) ──
    delta_tr  = sn_tr  - s_tr
    delta_val = sn_val - s_val
    delta_te  = sn_te  - s_te

    print(f"\nTrain      : {len(s_tr):,} transitions")
    print(f"Validation : {len(s_val):,} transitions")
    print(f"Test       : {len(s_te):,} transitions")
    print(f"Entrée DNN : {s_tr.shape[1]}D état + {a_tr.shape[1]}D action = 6D")
    print(f"Cible DNN  : Δs ({delta_tr.shape[1]}D)  — prédit delta, pas s' directement")
    print(f"  |Δs| moyen (train) : {delta_tr.abs().mean():.5f}  "
          f"vs  |s'| moyen : {sn_tr.abs().mean():.5f}")

    return (s_tr, a_tr, delta_tr,
            s_val, a_val, delta_val,
            s_te, a_te, delta_te,
            sn_te,           # ← état suivant réel (pour évaluation physique)
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

    Architecture Small [64-64] — meilleure sur CartPole (MSE=0.000004, R²=1.0)
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
        x = torch.cat([state, action], dim=1)
        return self.net(x)

    def predict_next(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Reconstruit s_{t+1} = s_t + Δs prédit."""
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
      Si LOSS_TYPE="mse"   : loss = mean( (Δs_pred - Δs_réel)² )
      Si LOSS_TYPE="huber" : loss = Huber( Δs_pred - Δs_réel )
      Dans les deux cas, minimiser sur Δs est équivalent à minimiser
      sur s' car s_t est fixé, mais Δs a une amplitude plus petite
      → gradients plus stables, convergence plus rapide.
    """
    stopper   = EarlyStopping()
    best_val  = float("inf")
    tr_losses, val_losses = [], []
    t0 = time.time()

    for epoch in range(EPOCHS):

        # ── Train ──
        model.train()
        tr_loss = 0.0
        for s, a, delta in train_loader:
            s, a, delta = s.to(DEVICE), a.to(DEVICE), delta.to(DEVICE)
            pred = model(s, a)              # prédit Δs
            loss = criterion(pred, delta)   # compare Δs_pred vs Δs_réel
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * s.size(0)
        tr_loss /= len(train_loader.dataset)  # moyenne par transition (pas par batch)

        # ── Validation ──
        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for s, a, delta in val_loader:
                s, a, delta = s.to(DEVICE), a.to(DEVICE), delta.to(DEVICE)
                vl_loss += (criterion(model(s, a), delta).item() * s.size(0))
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
       (MSE ou Huber selon LOSS_TYPE)

    2. Sur s' physique (après reconstruction + dénormalisation)
       → interprétable en unités réelles (m, m/s, rad, rad/s)
       → c'est cette MSE qui va dans votre rapport

    Reconstruction :
        Δs_pred_norm = model(s_norm, a)
        s'_norm      = s_norm + Δs_pred_norm
        s'_phys      = denormalize(s'_norm)
    """
    criterion_eval = get_loss_fn(LOSS_TYPE, HUBER_DELTA)
    model.eval()

    # ── Métriques sur Δs ──
    deltas_pred, deltas_real = [], []
    with torch.no_grad():
        for s, a, delta in test_loader:
            s, a = s.to(DEVICE), a.to(DEVICE)
            deltas_pred.append(model(s, a).cpu().numpy())
            deltas_real.append(delta.numpy())

    dp = np.concatenate(deltas_pred)
    dr = np.concatenate(deltas_real)

    loss_delta = float(np.mean(
        (dp - dr)**2 if LOSS_TYPE == "mse"
        else np.where(np.abs(dp-dr) < HUBER_DELTA,
                      0.5*(dp-dr)**2,
                      HUBER_DELTA*(np.abs(dp-dr) - 0.5*HUBER_DELTA))
    ))
    # loss_delta = criterion_eval(
    #                 torch.tensor(dp),
    #                 torch.tensor(dr)
    #             ).item()

    # ── Reconstruction s' et métriques physiques ──
    s_norm  = s_te_tensor.numpy()                      # (N, 4) normalisé
    sn_norm = sn_te_tensor.numpy()                     # (N, 4) normalisé (réel)

    spred_norm = s_norm + dp                           # s' = s + Δs_prédit
    spred_phys = denormalize(spred_norm, mean, std)
    sreal_phys = denormalize(sn_norm,   mean, std)

    err_phys     = spred_phys - sreal_phys
    mse_per_dim  = np.mean(err_phys**2,        axis=0)
    mae_per_dim  = np.mean(np.abs(err_phys),   axis=0)
    rmse_per_dim = np.sqrt(mse_per_dim)

    ss_res = np.sum((sreal_phys - spred_phys)**2)
    ss_tot = np.sum((sreal_phys - sreal_phys.mean(0))**2)
    r2     = 1 - ss_res / (ss_tot + 1e-10)

    within = np.abs(spred_norm - sn_norm) < eps
    acc_all = within.all(axis=1).mean()

    dim_names = ["x (m)", "ẋ (m/s)", "θ (rad)", "θ̇ (rad/s)"]
    print(f"\n{'─'*62}")
    print(f"Loss {LOSS_TYPE.upper()} sur Δs (test)  : {loss_delta:.7f}")
    print(f"R²   sur s' physique      : {r2:.6f}")
    print(f"Accuracy ε={eps} (norm.)  : {acc_all:.4f}")
    print(f"\n{'Dim':12s} {'MSE':>12} {'RMSE':>12} {'MAE':>12}")
    print(f"{'─'*50}")
    for i, name in enumerate(dim_names):
        print(f"{name:12s} {mse_per_dim[i]:>12.6f} "
              f"{rmse_per_dim[i]:>12.6f} {mae_per_dim[i]:>12.6f}")
    print(f"{'─'*62}")
    model.train()

    return spred_phys, sreal_phys, {
        "loss_delta": loss_delta,
        "r2": r2,
        "acc": acc_all,
        "mse_per_dim": mse_per_dim,
        "rmse_per_dim": rmse_per_dim,
        "mae_per_dim": mae_per_dim,
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
    plt.savefig("training_curve.png", dpi=130)
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
    plt.savefig("predictions_scatter.png", dpi=130)
    plt.show()
    print("Sauvegardé → predictions_scatter.png")


def plot_delta_distribution(s_te, delta_pred_norm, delta_real_norm):
    """
    Vérifie que les Δs prédits suivent la même distribution que les Δs réels.
    Histogramme superposé par dimension.
    Important pour le rapport : montre que le world model capture
    la dynamique réelle et non une approximation biaisée.
    """
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
    plt.savefig("delta_distribution.png", dpi=130)
    plt.show()
    print("Sauvegardé → delta_distribution.png")


def plot_rollout(model, s0_norm: np.ndarray, actions: list,
                 mean: np.ndarray, std: np.ndarray,
                 real_states_phys: np.ndarray):
    """
    Rollout séquentiel — prédictions enchaînées sur T pas.

    À chaque pas :
        Δs_pred  = model(s_norm, a_oh)
        s'_norm  = s_norm + Δs_pred      ← reconstruction via Δs
        s'_phys  = denormalize(s'_norm)

    Avantage du Δs ici : l'erreur s'accumule moins vite car
    le réseau prédit une petite correction, pas une valeur absolue.
    """
    model.eval()
    predicted_phys = []
    s = torch.tensor(s0_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        for a in actions:
            a_oh = torch.zeros(1, 2, device=DEVICE)
            a_oh[0, a] = 1.0
            s = model.predict_next(s, a_oh)   # s + Δs
            s_phys = denormalize(s.cpu().numpy(), mean, std)
            predicted_phys.append(s_phys.squeeze())

    predicted_phys = np.array(predicted_phys)
    T = min(len(predicted_phys), len(real_states_phys))

    dim_names = ["x (m)", "ẋ (m/s)", "θ (rad)", "θ̇ (rad/s)"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    axes = axes.flatten()

    for i, (ax, name) in enumerate(zip(axes, dim_names)):
        ax.plot(real_states_phys[:T, i],      color="#1D9E75",
                linewidth=2,   label="Réel")
        ax.plot(predicted_phys[:T, i],        color="#D85A30",
                linewidth=1.5, linestyle="--", label="Prédit (via Δs)")
        err = np.abs(predicted_phys[:T, i] - real_states_phys[:T, i])
        ax.fill_between(range(T),
                        real_states_phys[:T, i] - err,
                        real_states_phys[:T, i] + err,
                        alpha=0.12, color="#D85A30")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Pas de temps")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle(f"Rollout séquentiel — Predict Δs  [{LOSS_TYPE.upper()}]",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig("rollout.png", dpi=130)
    plt.show()
    print("Sauvegardé → rollout.png")


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # 1. Chargement
    (s_tr, a_tr, delta_tr,
     s_val, a_val, delta_val,
     s_te, a_te, delta_te,
     sn_te,
     mean, std) = load_data(DATA_PATH)
    
    std = np.where(std < 1e-8, 1.0, std)

    # 2. DataLoaders
    train_loader = DataLoader(TensorDataset(s_tr,  a_tr,  delta_tr),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(s_val, a_val, delta_val),
                              batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(TensorDataset(s_te,  a_te,  delta_te),
                              batch_size=BATCH_SIZE, shuffle=False)

    # 3. Modèle
    model = TransitionDNN().to(DEVICE)
    print(f"\nArchitecture :\n{model}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Paramètres   : {n_params:,}\n")

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

    # 7. Visualisation courbe d'entraînement
    plot_training_curve(tr_losses, val_losses)

    # 8. Évaluation sur le test set
    pred_phys, real_phys, metrics = evaluate(
        model, test_loader, s_te, sn_te, mean, std
    )
    plot_predictions(pred_phys, real_phys)

    # 9. Distribution des deltas (pour le rapport)
    model.eval()
    dp_list = []
    with torch.no_grad():
        for s, a, _ in test_loader:
            dp_list.append(model(s.to(DEVICE), a.to(DEVICE)).cpu().numpy())
    delta_pred_norm = np.concatenate(dp_list)
    delta_real_norm = delta_te.numpy()
    plot_delta_distribution(s_te.numpy(), delta_pred_norm, delta_real_norm)

    # 10. Rollout séquentiel
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=77)
    rollout_actions = [env.action_space.sample() for _ in range(60)]
    rollout_real    = [obs]
    for a in rollout_actions:
        obs, _, term, trunc, _ = env.step(a)
        rollout_real.append(obs)
        if term or trunc:
            break
    env.close()

    rollout_real  = np.array(rollout_real, dtype=np.float32)
    s0_norm       = (rollout_real[0:1] - mean) / std

    plot_rollout(model, s0_norm.squeeze(), rollout_actions,
                 mean, std, rollout_real[1:])

    # 11. Sauvegarde finale
    torch.save(model.state_dict(), "transition_dnn_final.pth")
    print("\nModèle sauvegardé → transition_dnn_final.pth")
    print(f"\nRésumé final :")
    print(f"  Loss type    : {LOSS_TYPE.upper()}")
    print(f"  R²           : {metrics['r2']:.6f}")
    print(f"  Accuracy ε   : {metrics['acc']:.4f}")
    print(f"  RMSE physique: {metrics['rmse_per_dim'].mean():.6f}")