"""
dnn_compare_robust.py — Comparaison robuste d'architectures DNN
Objectif : trouver la meilleure architecture pour prédire Δs sur CartPole-v1

Chaque modèle est entraîné N_SEEDS fois → moyenne ± std pour éliminer
la variance d'initialisation aléatoire.

Métriques comparées :
  - Val loss (MSE sur Δs normalisé)
  - RMSE physique (en unités réelles : m, m/s, rad, rad/s)
  - R² sur s' reconstruit
  - Accuracy ε (tolérance normalisée)
  - Époques avant early stopping
  - Temps d'entraînement

Usage :
    python dnn_compare_robust.py
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════

DATA_PATH      = "cartpole_data_mixed_policy.npz"
CHECKPOINT_DIR = "checkpoints_compare"
PLOTS_DIR = "compare_plots"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE  = 256
EPOCHS      = 200
LR          = 1e-3
PATIENCE    = 15
MIN_DELTA   = 1e-6
N_SEEDS     = 5          # runs par architecture (3 minimum, 5 recommandé)
EPS         = 0.01       # tolérance normalisée pour accuracy

# ==== LOSS CHOICE ====
LOSS_TYPE   = "mse"      # "mse" ou "huber" — MSE recommandé (voir analyse)
HUBER_DELTA = 1.0

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
print(f"Device : {DEVICE}  |  Loss : {LOSS_TYPE.upper()}  |  {N_SEEDS} seeds/architecture")


# ════════════════════════════════════════════════════════════
# REPRODUCTIBILITÉ
# ════════════════════════════════════════════════════════════

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ════════════════════════════════════════════════════════════
# LOSS FACTORY
# ════════════════════════════════════════════════════════════

def get_loss_fn(loss_type: str = LOSS_TYPE, delta: float = HUBER_DELTA) -> nn.Module:
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "huber":
        return nn.HuberLoss(delta=delta)
    else:
        raise ValueError(f"loss_type doit être 'mse' ou 'huber'. Reçu : '{loss_type}'")


# ════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES
# ════════════════════════════════════════════════════════════

def load_data(path: str):
    """
    Charge le NPZ et construit Δs = sn_norm - s_norm.
    Même logique que train_dnn_advanced.py pour cohérence.
    """
    data = np.load(path)
    def t(k): return torch.tensor(data[k], dtype=torch.float32)

    s_tr,  a_tr,  sn_tr  = t("s_train"), t("a_train"), t("sn_train")
    s_val, a_val, sn_val = t("s_val"),   t("a_val"),   t("sn_val")
    s_te,  a_te,  sn_te  = t("s_test"),  t("a_test"),  t("sn_test")

    mean = data["mean"].astype(np.float32)
    std  = data["std"].astype(np.float32)

    # Cibles : Δs dans l'espace normalisé
    delta_tr  = sn_tr  - s_tr
    delta_val = sn_val - s_val
    delta_te  = sn_te  - s_te

    print(f"\nTrain={len(s_tr):,}  Val={len(s_val):,}  Test={len(s_te):,}")
    print(f"|Δs| moyen = {delta_tr.abs().mean():.5f}  "
          f"|s'| moyen = {sn_tr.abs().mean():.5f}")

    return (s_tr, a_tr, delta_tr,
            s_val, a_val, delta_val,
            s_te,  a_te,  delta_te, sn_te,
            mean, std)


# ════════════════════════════════════════════════════════════
# DÉNORMALISATION
# ════════════════════════════════════════════════════════════

def denormalize(s_norm: np.ndarray, mean, std) -> np.ndarray:
    return s_norm * std + mean


# ════════════════════════════════════════════════════════════
# FACTORY D'ARCHITECTURES
# ════════════════════════════════════════════════════════════

def build_mlp(layers: list[int],
              activation: str = "relu",
              batchnorm: bool = False,
              dropout: float = 0.0,
              state_dim: int = 4,
              action_dim: int = 2) -> nn.Sequential:
    """
    Construit un MLP générique pour prédire Δs.

    Entrée  : state_dim + action_dim = 6D
    Sortie  : state_dim = 4D (Δs)

    Args:
        layers     : liste des tailles de couches cachées, ex [64, 64]
        activation : "relu" ou "leakyrelu"
        batchnorm  : True pour ajouter BatchNorm1d après chaque couche
        dropout    : taux de dropout (0.0 = désactivé)
    """
    act_map = {
        "relu":      nn.ReLU,
        "leakyrelu": lambda: nn.LeakyReLU(negative_slope=0.1),
        "elu":       nn.ELU,
        "tanh":      nn.Tanh,
    }
    assert activation in act_map, f"Activation inconnue : {activation}"

    modules = []
    in_dim  = state_dim + action_dim

    for h in layers:
        modules.append(nn.Linear(in_dim, h))
        if batchnorm:
            modules.append(nn.BatchNorm1d(h))
        modules.append(act_map[activation]())
        if dropout > 0.0:
            modules.append(nn.Dropout(dropout))
        in_dim = h

    modules.append(nn.Linear(in_dim, state_dim))
    return nn.Sequential(*modules)


class TransitionDNN(nn.Module):
    """World model générique — prédit Δs = s_{t+1} - s_t."""
    def __init__(self, net: nn.Sequential):
        super().__init__()
        self.net = net

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=1))

    def predict_next(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Reconstruit s_{t+1} = s_t + Δs prédit."""
        return state + self.forward(state, action)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# Catalogue des architectures à comparer
# Clé = nom affiché  |  Valeur = lambda qui construit le réseau
MODEL_ZOO = {
    # ── Largeur ──
    "Tiny      [32-32]":          lambda: build_mlp([32, 32]),
    "Small     [64-64]":          lambda: build_mlp([64, 64]),
    "Medium    [128-128]":        lambda: build_mlp([128, 128]),
    "Large     [256-256]":        lambda: build_mlp([256, 256]),
    # ── Profondeur ──
    "Deep3     [64-64-64]":       lambda: build_mlp([64, 64, 64]),
    "Deep4     [128-128-64-32]":  lambda: build_mlp([128, 128, 64, 32]),
    "Deep5     [256-256-128-64-32]": lambda: build_mlp([256, 256, 128, 64, 32]),
    # ── Entonnoir ──
    "Funnel    [128-64-32]":      lambda: build_mlp([128, 64, 32]),
    "Funnel2   [256-128-64]":     lambda: build_mlp([256, 128, 64]),
    # ── Activations ──
    "LeakyReLU [64-64]":          lambda: build_mlp([64, 64], activation="leakyrelu"),
    "ELU       [64-64]":          lambda: build_mlp([64, 64], activation="elu"),
    "Tanh      [64-64]":          lambda: build_mlp([64, 64], activation="tanh"),
    # ── Régularisation ──
    "BatchNorm [128-128]":        lambda: build_mlp([128, 128], batchnorm=True),
    "Dropout   [128-128]":        lambda: build_mlp([128, 128], dropout=0.1),
}


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
# UN SEUL RUN D'ENTRAÎNEMENT
# ════════════════════════════════════════════════════════════

def run_once(model: TransitionDNN,
             train_loader: DataLoader,
             val_loader: DataLoader,
             ckpt_path: str) -> tuple[float, int, float]:
    """
    Entraîne le modèle une fois, sauvegarde le meilleur checkpoint.

    Retourne (best_val_loss, n_epochs, training_time_s).
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = get_loss_fn()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    stopper   = EarlyStopping()
    best_val  = float("inf")
    n_epochs  = 0
    t0        = time.time()

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        tr_loss = 0.0
        for s, a, delta in train_loader:
            s, a, delta = s.to(DEVICE), a.to(DEVICE), delta.to(DEVICE)
            loss = criterion(model(s, a), delta)
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
        n_epochs = epoch + 1

        if vl_loss < best_val:
            best_val = vl_loss
            torch.save(model.state_dict(), ckpt_path)

        if stopper.step(vl_loss):
            print(f"    early stop @ epoch {epoch+1}  best_val={best_val:.6f}")
            break

    return best_val, n_epochs, time.time() - t0


# ════════════════════════════════════════════════════════════
# ÉVALUATION SUR LE TEST SET
# ════════════════════════════════════════════════════════════

def evaluate_model(model: TransitionDNN,
                   test_loader: DataLoader,
                   s_te: torch.Tensor,
                   sn_te: torch.Tensor,
                   mean: np.ndarray,
                   std: np.ndarray) -> dict:
    """
    Métriques complètes sur le test set :
      - val_loss (MSE/Huber sur Δs normalisé)
      - RMSE physique globale
      - RMSE par dimension
      - R²
      - Accuracy ε
    """
    criterion = get_loss_fn()
    model.eval()

    dp_list, dr_list = [], []
    with torch.no_grad():
        for s, a, delta in test_loader:
            s, a = s.to(DEVICE), a.to(DEVICE)
            dp_list.append(model(s, a).cpu().numpy())
            dr_list.append(delta.numpy())

    dp = np.concatenate(dp_list)   # Δs prédit (normalisé)
    dr = np.concatenate(dr_list)   # Δs réel   (normalisé)

    # Loss sur Δs
    test_loss = float(np.mean((dp - dr) ** 2))   # MSE toujours pour comparaison équitable

    # Reconstruction s' et métriques physiques
    s_norm      = s_te.numpy()
    sn_norm     = sn_te.numpy()
    spred_norm  = s_norm + dp
    spred_phys  = denormalize(spred_norm, mean, std)
    sreal_phys  = denormalize(sn_norm,    mean, std)

    err         = spred_phys - sreal_phys
    mse_per_dim = np.mean(err ** 2, axis=0)
    rmse_phys   = float(np.sqrt(np.mean(err ** 2)))
    mae_phys    = float(np.mean(np.abs(err)))

    ss_res      = np.sum((sreal_phys - spred_phys) ** 2)
    ss_tot      = np.sum((sreal_phys - sreal_phys.mean(0)) ** 2)
    r2          = float(1 - ss_res / (ss_tot + 1e-12))

    within      = np.abs(spred_norm - sn_norm) < EPS
    acc         = float(within.all(axis=1).mean())

    return {
        "test_loss":   test_loss,
        "rmse_phys":   rmse_phys,
        "mae_phys":    mae_phys,
        "r2":          r2,
        "acc":         acc,
        "mse_per_dim": mse_per_dim,
    }


# ════════════════════════════════════════════════════════════
# VISUALISATIONS
# ════════════════════════════════════════════════════════════

COLORS = [
    "#378ADD", "#1D9E75", "#D85A30", "#9B59B6",
    "#E67E22", "#2ECC71", "#E74C3C", "#1ABC9C",
    "#3498DB", "#F39C12", "#8E44AD", "#27AE60",
    "#E91E63", "#00BCD4",
]


def plot_barplot(summary: dict, metric: str, ylabel: str, title: str, fname: str):
    """Barplot avec barres d'erreur + points individuels (jitter)."""
    names  = list(summary.keys())
    means  = [summary[n][f"{metric}_mean"] for n in names]
    stds   = [summary[n][f"{metric}_std"]  for n in names]
    all_v  = [summary[n][f"{metric}_all"]  for n in names]

    # Trier par moyenne croissante
    order  = np.argsort(means)
    names  = [names[i]  for i in order]
    means  = [means[i]  for i in order]
    stds   = [stds[i]   for i in order]
    all_v  = [all_v[i]  for i in order]

    x   = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(12, len(names)*1.1), 5))

    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=[COLORS[i % len(COLORS)] for i in range(len(names))],
                  alpha=0.75, width=0.55,
                  error_kw=dict(ecolor="black", elinewidth=1.2, capthick=1.2))

    # Points individuels (jitter)
    for i, vals in enumerate(all_v):
        jitter = np.random.uniform(-0.18, 0.18, len(vals))
        ax.scatter(i + jitter, vals, color="black",
                   s=20, zorder=5, alpha=0.6)

    # Valeur au-dessus de chaque barre
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + max(means) * 0.01,
                f"{mean:.2e}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"{title}\n({N_SEEDS} seeds — barres = ±std, points = runs individuels)",
                 fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=140)
    plt.show()
    print(f"Sauvegardé → {fname}")


def plot_radar(summary: dict, top_n: int = 6):
    """
    Radar chart pour les top_n modèles sur 5 métriques normalisées.
    Permet de voir les trade-offs entre performance, stabilité, et efficacité.
    """
    metrics_cfg = [
        ("test_loss_mean",  "MSE Δs",      True),   # True = lower is better
        ("rmse_phys_mean",  "RMSE phys.",  True),
        ("r2_mean",         "R²",          False),
        ("acc_mean",        "Accuracy ε",  False),
        ("epochs_mean",     "Vitesse\n(1/epochs)", True),
    ]

    names  = list(summary.keys())
    # Sélectionner top_n selon test_loss
    sorted_names = sorted(names, key=lambda n: summary[n]["test_loss_mean"])[:top_n]

    angles = np.linspace(0, 2 * np.pi, len(metrics_cfg), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for idx, name in enumerate(sorted_names):
        values = []
        for key, _, lower_better in metrics_cfg:
            raw_vals = [summary[n][key] for n in names]
            v        = summary[name][key]
            lo, hi   = min(raw_vals), max(raw_vals)
            if hi == lo:
                norm = 1.0
            elif lower_better:
                norm = 1.0 - (v - lo) / (hi - lo)   # inversé : plus bas = meilleur
            else:
                norm = (v - lo) / (hi - lo)
            values.append(norm)
        values += values[:1]

        color = COLORS[idx % len(COLORS)]
        ax.plot(angles, values, color=color, linewidth=2, label=name.strip())
        ax.fill(angles, values, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m[1] for m in metrics_cfg], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(f"Radar — Top {top_n} architectures\n(normalisé 0→1, vers l'extérieur = meilleur)",
                 fontsize=11, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "radar_comparison.png"), dpi=140)
    plt.show()
    print("Sauvegardé → " + os.path.join(PLOTS_DIR, "radar_comparison.png"))


def plot_rmse_per_dim(summary: dict, top_n: int = 5):
    """
    Heatmap RMSE physique par dimension pour les top_n modèles.
    Utile pour voir si un modèle est bon sur θ mais mauvais sur x.
    """
    dim_names  = ["x (m)", "ẋ (m/s)", "θ (rad)", "θ̇ (rad/s)"]
    names      = list(summary.keys())
    sorted_names = sorted(names, key=lambda n: summary[n]["rmse_phys_mean"])[:top_n]

    matrix = np.array([
        np.sqrt(summary[n]["mse_dim_mean"])
        for n in sorted_names
    ])  # (top_n, 4)

    fig, ax = plt.subplots(figsize=(9, max(3, top_n * 0.7)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(4))
    ax.set_xticklabels(dim_names, fontsize=10)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([n.strip() for n in sorted_names], fontsize=9)
    plt.colorbar(im, ax=ax, label="RMSE (unités physiques)")

    for i in range(top_n):
        for j in range(4):
            ax.text(j, i, f"{matrix[i,j]:.4f}",
                    ha="center", va="center", fontsize=8,
                    color="black" if matrix[i,j] < matrix.max()*0.6 else "white")

    ax.set_title(f"RMSE physique par dimension — Top {top_n} architectures",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "rmse_per_dim_heatmap.png"), dpi=140)
    plt.show()
    print("Sauvegardé → " + os.path.join(PLOTS_DIR, "rmse_per_dim_heatmap.png"))


def print_final_table(summary: dict):
    """Tableau récapitulatif console trié par test_loss."""
    col = 26
    header = (f"{'Architecture':<{col}} {'MSE_Δs':>12} {'±std':>10} "
              f"{'RMSE_phys':>12} {'R²':>8} {'Acc_ε':>8} "
              f"{'Epochs':>8} {'Params':>8}")
    sep = "═" * len(header)
    print(f"\n{sep}")
    print(header)
    print("─" * len(header))

    sorted_names = sorted(summary.keys(),
                          key=lambda n: summary[n]["test_loss_mean"])

    for rank, name in enumerate(sorted_names, 1):
        m = summary[name]
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "  ")
        print(
            f"{medal} {name:<{col-2}} "
            f"{m['test_loss_mean']:>12.7f} "
            f"{m['test_loss_std']:>10.7f} "
            f"{m['rmse_phys_mean']:>12.6f} "
            f"{m['r2_mean']:>8.4f} "
            f"{m['acc_mean']:>8.4f} "
            f"{m['epochs_mean']:>8.1f} "
            f"{m['n_params']:>8,}"
        )
    print(sep)

    best = sorted_names[0]
    print(f"\n✓ Meilleure architecture : {best.strip()}")
    print(f"  MSE Δs  = {summary[best]['test_loss_mean']:.7f} "
          f"± {summary[best]['test_loss_std']:.7f}")
    print(f"  RMSE phys = {summary[best]['rmse_phys_mean']:.6f}")
    print(f"  R²        = {summary[best]['r2_mean']:.6f}")
    return best


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # 1. Chargement
    (s_tr, a_tr, delta_tr,
     s_val, a_val, delta_val,
     s_te, a_te, delta_te, sn_te,
     mean, std) = load_data(DATA_PATH)

    train_loader = DataLoader(TensorDataset(s_tr,  a_tr,  delta_tr),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(s_val, a_val, delta_val),
                              batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(TensorDataset(s_te,  a_te,  delta_te),
                              batch_size=BATCH_SIZE, shuffle=False)

    # 2. Boucle de comparaison
    summary = {}

    for model_name, build_fn in MODEL_ZOO.items():
        print(f"\n{'═'*60}")
        print(f"  {model_name.strip()}  ({N_SEEDS} seeds)")
        print(f"{'═'*60}")

        seed_results = []
        n_params     = None

        for seed in range(N_SEEDS):
            set_seed(seed)

            net   = build_fn()
            model = TransitionDNN(net).to(DEVICE)

            if n_params is None:
                n_params = model.count_params()

            slug  = model_name.split()[0]
            ckpt  = f"{CHECKPOINT_DIR}/{slug}_seed{seed}.pth"

            print(f"  seed {seed} ...", end=" ", flush=True)
            best_val, n_epochs, t_train = run_once(
                model, train_loader, val_loader, ckpt
            )

            # Charger le meilleur checkpoint avant d'évaluer
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            metrics = evaluate_model(model, test_loader, s_te, sn_te, mean, std)
            metrics["val_loss"] = best_val
            metrics["n_epochs"] = n_epochs
            metrics["t_train"]  = t_train
            seed_results.append(metrics)

            print(f"MSE={metrics['test_loss']:.7f}  "
                  f"RMSE={metrics['rmse_phys']:.6f}  "
                  f"R²={metrics['r2']:.4f}  "
                  f"epochs={n_epochs}")

        # Agrégation des N seeds
        def agg(key):
            vals = [r[key] for r in seed_results]
            return np.mean(vals), np.std(vals), vals

        tl_m, tl_s, tl_a = agg("test_loss")
        rp_m, rp_s, rp_a = agg("rmse_phys")
        r2_m, r2_s, r2_a = agg("r2")
        ac_m, ac_s, ac_a = agg("acc")
        ep_m, ep_s, ep_a = agg("n_epochs")
        mse_dim_mean      = np.mean([r["mse_per_dim"] for r in seed_results], axis=0)

        summary[model_name] = {
            "test_loss_mean": tl_m, "test_loss_std": tl_s, "test_loss_all": tl_a,
            "rmse_phys_mean": rp_m, "rmse_phys_std": rp_s, "rmse_phys_all": rp_a,
            "r2_mean":        r2_m, "r2_std":        r2_s, "r2_all":        r2_a,
            "acc_mean":       ac_m, "acc_std":        ac_s, "acc_all":       ac_a,
            "epochs_mean":    ep_m, "epochs_std":     ep_s, "epochs_all":    ep_a,
            "mse_dim_mean":   mse_dim_mean,
            "n_params":       n_params,
        }

    # 3. Résultats
    best_name = print_final_table(summary)

    # 4. Visualisations
    plot_barplot(summary, "test_loss", "MSE sur Δs (normalisé)",
                 "Comparaison MSE Δs — toutes architectures",
                 "compare_mse_delta.png")

    plot_barplot(summary, "rmse_phys", "RMSE physique (m / m·s⁻¹ / rad)",
                 "Comparaison RMSE physique — toutes architectures",
                 "compare_rmse_phys.png")

    plot_radar(summary, top_n=6)
    plot_rmse_per_dim(summary, top_n=5)

    # 5. Sauvegarde des résultats bruts
    np.save(os.path.join(PLOTS_DIR, "comparison_summary.npy"), summary)
    print("\nRésultats sauvegardés → " + os.path.join(PLOTS_DIR, "comparison_summary.npy"))
    print("\nFigures générées :")
    print("  compare_mse_delta.png     — MSE sur Δs toutes architectures")
    print("  compare_rmse_phys.png     — RMSE physique toutes architectures")
    print("  radar_comparison.png      — Radar top 6 (multi-critères)")
    print("  rmse_per_dim_heatmap.png  — RMSE par dimension top 5")