"""
data_collector.py
=================
Collecte des transitions (s, a, r, s', done) pour entraîner le World Model.

Politique mixte par environnement :
  CartPole-v1    : 60% aléatoire + 40% heuristique (angle-based)
  MountainCar-v0 : 50% aléatoire + 50% heuristique (velocity-based)
                   + bonus : politique "swing" pour atteindre le sommet
  LunarLander-v3 : 60% aléatoire + 40% heuristique (altitude-based)

Pourquoi une politique mixte ?
  Une politique 100% aléatoire ne couvre pas les états importants :
  - MountainCar  : n'atteint jamais le sommet → WM n'apprend pas le goal
  - LunarLander  : ne se stabilise jamais    → WM n'apprend pas l'atterrissage
  La politique mixte force la couverture de ces états critiques.

Sorties :
  data/data_CartPole-v1.npz
  data/data_MountainCar-v0.npz
  data/data_LunarLander-v3.npz

  Chaque .npz contient :
    states      : (N, obs_dim)   float32
    actions     : (N,)           int32
    rewards     : (N,)           float32
    next_states : (N, obs_dim)   float32
    dones       : (N,)           float32  (0.0 ou 1.0)
    mean        : (obs_dim,)     float32  — stats de normalisation
    std         : (obs_dim,)     float32

Usage :
  # Collecter pour un seul env
  python ./collect_data/data_collect_general.py --env CartPole-v1
  python ./collect_data/data_collect_general.py --env MountainCar-v0 --steps 50000
  python ./collect_data/data_collect_general.py --env LunarLander-v3 --force

  # Collecter pour tous les envs d'un coup
  python ./collect_data/data_collect_general.py --all

  # Vérifier un fichier existant
  python ./collect_data/data_collect_general.py --inspect data/data_CartPole-v1.npz

Import dans world_model.py :
  from data_collector import load_dataset
  transitions = load_dataset("data/data_CartPole-v1.npz")
"""

import argparse
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
BASE_DIR  = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

SEED = 42

# Paramètres de collecte par environnement
ENV_CONFIG = {
    "CartPole-v1": {
        "steps":        100_000,
        "random_ratio": 0.60,      # 60% aléatoire
        "max_ep_steps": 500,
    },
    "MountainCar-v0": {
        "steps":        100_000,    # plus de steps car env difficile
        "random_ratio": 0.50,
        "max_ep_steps": 200,
    },
    "LunarLander-v3": {
        "steps":        100_000,    # env complexe (8D state)
        "random_ratio": 0.60,
        "max_ep_steps": 1000,
    },
}


# ════════════════════════════════════════════════════════════
#  POLITIQUES HEURISTIQUES
# ════════════════════════════════════════════════════════════

def heuristic_cartpole(obs: np.ndarray) -> int:
    """
    Pousse dans la direction opposée à l'inclinaison du pendule.
    obs : [x, x_dot, theta, theta_dot]
    """
    theta     = obs[2]
    theta_dot = obs[3]
    # Si le pendule penche à droite → pousse à droite
    return 1 if (theta + 0.1 * theta_dot) > 0 else 0


def heuristic_mountaincar(obs: np.ndarray) -> int:
    """
    Politique "swing" : pousse dans le sens de la vitesse actuelle.
    obs : [position, velocity]
    Stratégie : accumuler de l'élan en oscillant
    action : 0=gauche, 1=rien, 2=droite
    """
    position = obs[0]
    velocity = obs[1]

    # Si on monte vers la droite → continuer
    if velocity > 0:
        return 2
    # Si on descend vers la gauche → continuer pour accumuler l'élan
    elif velocity < 0:
        return 0
    # Neutre
    else:
        return 2 if position < -0.5 else 0


def heuristic_lunarlander(obs: np.ndarray) -> int:
    """
    Politique basique d'atterrissage.
    obs : [x, y, vx, vy, angle, angular_vel, leg_left, leg_right]
    action : 0=rien, 1=moteur_gauche, 2=moteur_principal, 3=moteur_droite
    """
    x         = obs[0]   # position horizontale (0 = centre)
    y         = obs[1]   # altitude
    vx        = obs[2]   # vitesse horizontale
    vy        = obs[3]   # vitesse verticale
    angle     = obs[4]   # inclinaison

    # Freiner la descente si on tombe trop vite
    if vy < -0.5 and y > 0.3:
        return 2    # moteur principal

    # Corriger l'inclinaison
    if angle > 0.1:
        return 3    # moteur droite pour corriger
    if angle < -0.1:
        return 1    # moteur gauche

    # Corriger la dérive horizontale
    if x > 0.2:
        return 1
    if x < -0.2:
        return 3

    # Par défaut : freiner légèrement
    if vy < -0.3:
        return 2

    return 0    # rien


# Map env → heuristique
HEURISTICS = {
    "CartPole-v1":    heuristic_cartpole,
    "MountainCar-v0": heuristic_mountaincar,
    "LunarLander-v3": heuristic_lunarlander,
}


def split_dataset(data, train_ratio=0.7, val_ratio=0.1, seed=42):
    N = data["states"].shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    tr = int(train_ratio * N)
    val = int(val_ratio * N)
    train_idx = idx[:tr]
    val_idx   = idx[tr:tr+val]
    test_idx  = idx[tr+val:]

    def split_arr(arr):
        return arr[train_idx], arr[val_idx], arr[test_idx]
    
    res = {}
    for k in ["states", "actions", "rewards", "next_states", "dones"]:
        res[f"{k}_train"], res[f"{k}_val"], res[f"{k}_test"] = split_arr(data[k])
    return res


# ════════════════════════════════════════════════════════════
#  COLLECTE PRINCIPALE
# ════════════════════════════════════════════════════════════

def collect(env_name: str, n_steps: int = None,
            random_ratio: float = None,
            seed: int = SEED,
            verbose: bool = True) -> dict:
    """
    Collecte n_steps transitions sur env_name avec politique mixte.

    Retourne un dict avec clés :
        states, actions, rewards, next_states, dones
        + métadonnées : env_name, n_steps, episode_returns
    """
    cfg          = ENV_CONFIG.get(env_name, {})
    n_steps      = n_steps      or cfg.get("steps",        30_000)
    random_ratio = random_ratio or cfg.get("random_ratio", 0.60)
    heuristic    = HEURISTICS.get(env_name)

    try:
        env = gym.make(env_name)
    except Exception as e:
        raise ValueError(
            f"Erreur avec env={env_name}. Vérifie la version Gymnasium installée. Détail: {e}"
        )
    env.reset(seed=seed)
    np.random.seed(seed)

    obs_dim = env.observation_space.shape[0]
    n_act   = env.action_space.n

    # Pré-alloue les arrays → beaucoup plus rapide que des listes
    states      = np.zeros((n_steps, obs_dim), dtype=np.float32)
    actions     = np.zeros(n_steps,            dtype=np.int32)
    rewards     = np.zeros(n_steps,            dtype=np.float32)
    next_states = np.zeros((n_steps, obs_dim), dtype=np.float32)
    dones       = np.zeros(n_steps,            dtype=np.float32)

    # Statistiques de monitoring
    episode_returns  = []
    ep_return        = 0.0
    ep_count         = 0
    goal_reached     = 0   # MountainCar uniquement

    obs, _ = env.reset()

    if verbose:
        print(f"\n[COLLECT] {env_name}")
        print(f"  Steps={n_steps} | random_ratio={random_ratio:.0%} "
              f"| heuristic={'oui' if heuristic else 'non'}")
        print(f"  obs_dim={obs_dim} | n_actions={n_act}\n")

    for i in range(n_steps):

        # ── Choix de l'action ────────────────────────────────
        if np.random.rand() < random_ratio or heuristic is None:
            action = env.action_space.sample()
        else:
            action = heuristic(obs)

        # ── Step ─────────────────────────────────────────────
        next_obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc

        # ── Stockage ─────────────────────────────────────────
        states[i]      = obs
        actions[i]     = action
        rewards[i]     = reward
        next_states[i] = next_obs
        dones[i]       = float(done)

        ep_return += reward

        # MountainCar : goal = position >= 0.5
        if env_name == "MountainCar-v0" and next_obs[0] >= 0.45:
            goal_reached += 1

        if done:
            episode_returns.append(ep_return)
            ep_return = 0.0
            ep_count += 1
            obs, _ = env.reset()
        else:
            obs = next_obs

        # Log tous les 10k steps
        if verbose and (i + 1) % 10_000 == 0:
            avg_ret = (np.mean(episode_returns[-20:])
                       if episode_returns else 0.0)
            print(f"  {i+1:6d}/{n_steps} steps | "
                  f"{ep_count} épisodes | "
                  f"avg_ret(20ep)={avg_ret:.1f}"
                  + (f" | goals={goal_reached}"
                     if env_name == "MountainCar-v0" else ""))

    env.close()

    if verbose:
        print(f"\n[COLLECT] Terminé.")
        print(f"  Total épisodes : {ep_count}")
        if episode_returns:
            print(f"  Return moyen   : {np.mean(episode_returns):.2f} "
                  f"± {np.std(episode_returns):.2f}")
            print(f"  Return max     : {np.max(episode_returns):.2f}")
        if env_name == "MountainCar-v0":
            print(f"  Steps près du goal (pos≥0.45) : {goal_reached}")
            if goal_reached < 10:
                print("  ⚠ Peu de goals atteints — le WM aura peu d'exemples "
                      "de transitions positives.")

    return {
        "env_name":        env_name,
        "states":          states,
        "actions":         actions,
        "rewards":         rewards,
        "next_states":     next_states,
        "dones":           dones,
        "episode_returns": np.array(episode_returns, dtype=np.float32),
        "n_steps":         n_steps,
    }


# ════════════════════════════════════════════════════════════
#  SAUVEGARDE / CHARGEMENT
# ════════════════════════════════════════════════════════════

def save_dataset(data: dict, path: str = None) -> str:
    """
    Sauvegarde le dataset + mean/std de normalisation dans un .npz.
    Retourne le chemin du fichier sauvegardé.
    """
    if path is None:
        path = str(DATA_DIR / f"data_{data['env_name']}.npz")

    # Calcul mean/std sur les états (utilisé par world_model.py)
    splits = split_dataset(data)
    mean = splits['states_train'].mean(axis=0).astype(np.float32)
    std  = (splits['states_train'].std(axis=0) + 1e-8).astype(np.float32)
    
    # mean = data["states"].mean(axis=0).astype(np.float32)
    # std  = (data["states"].std(axis=0) + 1e-8).astype(np.float32)


    np.savez(
        path,
        states_train      = splits['states_train'],
        actions_train     = splits['actions_train'],
        rewards_train     = splits['rewards_train'],
        next_states_train = splits['next_states_train'],
        dones_train       = splits['dones_train'],
        states_val      = splits['states_val'],
        actions_val     = splits['actions_val'],
        rewards_val     = splits['rewards_val'],
        next_states_val = splits['next_states_val'],
        dones_val       = splits['dones_val'],
        states_test      = splits['states_test'],
        actions_test     = splits['actions_test'],
        rewards_test     = splits['rewards_test'],
        next_states_test = splits['next_states_test'],
        dones_test       = splits['dones_test'],
        mean        = mean,
        std         = std,
        env_name    = np.array([data["env_name"]]),
    )

    print(f"\n[SAVE] Dataset sauvegardé → {path}")
    print(f"  Taille train : {data['states'].shape[0]:,} transitions")
    # print(f"  Taille val   : {data['states_val'].shape[0]:,} transitions")
    # print(f"  Taille test  : {data['states_test'].shape[0]:,} transitions")
    print(f"  mean   : {np.round(mean, 4)}")
    print(f"  std    : {np.round(std,  4)}")
    return path


def load_dataset(path: str) -> dict:
    """
    Charge un .npz et retourne un dict compatible avec world_model.py.

    Usage dans world_model.py :
        from data_collector import load_dataset
        data = load_dataset("data/data_CartPole-v1.npz")
        # data["states"], data["actions"], data["mean"], data["std"]...
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset non trouvé : {path}")

    npz  = np.load(path, allow_pickle=True)
    data = {
        "states":      npz["states"].astype(np.float32),
        "actions":     npz["actions"].astype(np.int32),
        "rewards":     npz["rewards"].astype(np.float32),
        "next_states": npz["next_states"].astype(np.float32),
        "dones":       npz["dones"].astype(np.float32),
        "mean":        npz["mean"].astype(np.float32),
        "std":         npz["std"].astype(np.float32),
        "env_name":    str(npz["env_name"][0]),
    }
    print(f"[LOAD] {path}")
    print(f"  {data['states'].shape[0]:,} transitions | "
          f"obs_dim={data['states'].shape[1]} | "
          f"env={data['env_name']}")
    return data


# ════════════════════════════════════════════════════════════
#  INSPECTION / VISUALISATION
# ════════════════════════════════════════════════════════════

def inspect_dataset(path: str):
    """
    Affiche des statistiques détaillées sur un dataset existant.
    Utile pour vérifier que la couverture est suffisante.
    """
    data = load_dataset(path)
    s    = data["states"]
    a    = data["actions"]
    r    = data["rewards"]
    d    = data["dones"]

    print(f"\n{'='*55}")
    print(f"  INSPECTION : {path}")
    print(f"{'='*55}")
    print(f"  Transitions  : {s.shape[0]:,}")
    print(f"  Épisodes     : {int(d.sum()):,}")
    print(f"  Obs dim      : {s.shape[1]}")
    print(f"\n  Distribution des actions :")
    n_act = int(a.max()) + 1
    for i in range(n_act):
        pct = 100 * (a == i).sum() / len(a)
        print(f"    action {i} : {pct:.1f}%  ({(a==i).sum():,} fois)")

    print(f"\n  Statistiques des états (par dimension) :")
    print(f"  {'dim':<5} {'min':>8} {'max':>8} {'mean':>8} {'std':>8}")
    for dim in range(s.shape[1]):
        print(f"  {dim:<5} {s[:,dim].min():8.3f} {s[:,dim].max():8.3f} "
              f"{s[:,dim].mean():8.3f} {s[:,dim].std():8.3f}")

    print(f"\n  Reward  : min={r.min():.2f}  max={r.max():.2f}  "
          f"mean={r.mean():.4f}")
    print(f"  Done    : {d.sum():.0f} terminaisons "
          f"({100*d.mean():.2f}% des steps)")

    # Distribution des états (histogramme dim 0 et 1)
    fig, axes = plt.subplots(1, min(s.shape[1], 4), figsize=(14, 3))
    if s.shape[1] == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.hist(s[:, i], bins=50, color="#457b9d", alpha=0.8, edgecolor="none")
        ax.set_title(f"Dim {i}", fontsize=10)
        ax.set_xlabel("Valeur")
        ax.grid(alpha=0.3)
    plt.suptitle(f"Distribution des états — {data['env_name']}", y=1.02)
    plt.tight_layout()
    hist_path = path.replace(".npz", "_inspect.png")
    plt.savefig(hist_path, dpi=100, bbox_inches="tight")
    print(f"\n  Histogramme sauvegardé → {hist_path}")
    plt.show()


# ════════════════════════════════════════════════════════════
#  PIPELINE COMPLET
# ════════════════════════════════════════════════════════════

def collect_and_save(env_name: str, n_steps: int = None,
                     random_ratio: float = None,
                     seed: int = SEED,
                     force: bool = False) -> str:
    """
    Collecte et sauvegarde si le fichier n'existe pas déjà.
    Retourne le chemin du fichier .npz.
    """
    save_path = str(DATA_DIR / f"data_{env_name}.npz")

    if os.path.exists(save_path) and not force:
        print(f"[SKIP] {save_path} existe déjà. "
              f"Utilise --force pour re-collecter.")
        return save_path

    data = collect(env_name, n_steps, random_ratio, seed)
    return save_dataset(data, save_path)


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Collecte de transitions pour le World Model")
    parser.add_argument("--env",    default="CartPole-v1",
                        choices=list(ENV_CONFIG.keys()))
    parser.add_argument("--steps",  type=int,   default=None,
                        help="Nombre de steps (défaut selon l'env)")
    parser.add_argument("--ratio",  type=float, default=None,
                        help="Ratio aléatoire (défaut selon l'env)")
    parser.add_argument("--seed",   type=int,   default=SEED)
    parser.add_argument("--all",    action="store_true",
                        help="Collecte pour tous les environnements")
    parser.add_argument("--force",  action="store_true",
                        help="Re-collecte même si le fichier existe")
    parser.add_argument("--inspect", default=None,
                        help="Chemin d'un .npz à inspecter")
    args = parser.parse_args()

    # Mode inspection
    if args.inspect:
        inspect_dataset(args.inspect)
        return

    # Collecte
    if args.all:
        print("Collecte pour tous les environnements...\n")
        for env_name in ENV_CONFIG:
            collect_and_save(env_name, seed=args.seed, force=args.force)
            print()
    else:
        collect_and_save(
            env_name     = args.env,
            n_steps      = args.steps,
            random_ratio = args.ratio,
            seed         = args.seed,
            force        = args.force,
        )

    print("\nPour utiliser les données dans world_model.py :")
    print("  python world_model.py --env CartPole-v1 --config 3 "
          "--data data/data_CartPole-v1.npz")


if __name__ == "__main__":
    main()