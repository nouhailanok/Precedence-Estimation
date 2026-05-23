import numpy as np
from precedence_env import PrecedenceEnv
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. HELPERS POUR L'ÉTAT ET LES ACTIONS
# ─────────────────────────────────────────────

def flatten_obs(obs: dict, num_tasks: int) -> np.ndarray:
    """
    Transforme le dictionnaire d'état en un seul vecteur 1D.
    Taille = (N*N) pour la matrice + N pour completed + N pour ready.
    Pour N=5, ça donne un vecteur de taille 35.
    """
    adj = obs["adj_matrix"].flatten()
    comp = obs["completed"].flatten()
    ready = obs["ready"].flatten()
    
    return np.concatenate([adj, comp, ready]).astype(np.float32)

def get_mixed_action(obs: dict, rng: np.random.Generator, epsilon: float = 0.5) -> int:
    """
    Politique Mixte : 50% Aléatoire (parmi les actions valides) / 50% Heuristique.
    """
    # MASQUAGE : On récupère la liste des indices où la tâche est "Prête" (valeur 1)
    ready_tasks = np.where(obs["ready"] == 1)[0]
    
    # Sécurité : s'il n'y a plus de tâches prêtes, on renvoie 0 
    # (ne devrait pas arriver si terminated est bien géré)
    if len(ready_tasks) == 0:
        return 0

    if rng.random() < epsilon:
        # 50% du temps : Exploration (Choix aléatoire mais LÉGAL)
        return int(rng.choice(ready_tasks))
    else:
        # 50% du temps : Exploitation (Heuristique)
        # Heuristique simple : On prend la première tâche prête de la liste 
        # (Dans un vrai projet, ce serait "la tâche la plus courte")
        return int(ready_tasks[0])


# ─────────────────────────────────────────────
# 2. LA COLLECTE DE DONNÉES
# ─────────────────────────────────────────────

def collect_data_precedence(num_tasks: int = 5, n_steps: int = 100_000, seed: int = 0):
    # Instanciation de ton environnement personnalisé
    env = PrecedenceEnv(num_tasks=num_tasks)
    rng = np.random.default_rng(seed)

    states, actions, next_states = [], [], []
    episode = 0
    
    # Reset renvoie (obs, info), on ne garde que obs
    obs, _ = env.reset(seed=seed)

    for step in range(n_steps):
        # 1. Choix de l'action via la politique mixte
        action = get_mixed_action(obs, rng, epsilon=0.5)

        # 2. Éxécution dans l'environnement
        next_obs, reward, terminated, truncated, _ = env.step(action)

        # 3. Sauvegarde (en aplatissant les dictionnaires)
        states.append(flatten_obs(obs, num_tasks))
        actions.append(action)
        next_states.append(flatten_obs(next_obs, num_tasks))

        # 4. Gestion de la fin d'épisode
        if terminated or truncated:
            episode += 1
            obs, _ = env.reset(seed=seed + episode)
        else:
            obs = next_obs

    # Conversion en tenseurs Numpy
    S = np.asarray(states, dtype=np.float32)
    A = np.asarray(actions, dtype=np.int64)
    SN = np.asarray(next_states, dtype=np.float32)

    print(f"Transitions collectées : {len(S):,} (épisodes terminés : {episode})")
    print(f"  states.shape      : {S.shape}   ← Entrée du réseau (ex: 35D pour 5 tâches)")
    print(f"  actions.shape     : {A.shape}   ← Indices de l'action choisie")
    print(f"  next_states.shape : {SN.shape}  ← Cible à prédire")
    
    return S, A, SN


# (On suppose que PrecedenceEnv, flatten_obs, get_mixed_action et collect_data_precedence 
# sont définis juste au-dessus dans ton script)

NUM_TASKS = 5  # À définir globalement pour la cohérence

# ─────────────────────────────────────────────
# 2. ANALYSE DU DATASET (Adapté pour Precedence)
# ─────────────────────────────────────────────

def analyze_dataset(S: np.ndarray, A: np.ndarray, SN: np.ndarray, num_tasks: int):
    """Affiche des statistiques adaptées aux matrices binaires de précédence."""
    print("\n--- Statistiques du Dataset Precedence ---")
    print(f"  Taille de l'état (S) : {S.shape[1]} variables (Graphe {num_tasks}x{num_tasks} + {num_tasks} Completed + {num_tasks} Ready)")
    
    # Analyse des actions
    action_counts = np.bincount(A.reshape(-1).astype(np.int64), minlength=num_tasks)
    print("\n--- Distribution des Actions ---")
    for i, count in enumerate(action_counts):
        print(f"  Tâche {i} exécutée : {count:,} fois ({count/len(A)*100:.1f}%)")

    # Visualisation des actions
    plt.figure(figsize=(6, 4))
    plt.bar(range(num_tasks), action_counts, color="#378ADD", alpha=0.8)
    plt.title("Distribution des tâches choisies")
    plt.xlabel("ID de la Tâche")
    plt.ylabel("Fréquence")
    plt.xticks(range(num_tasks))
    plt.tight_layout()
    plt.savefig("precedence_actions_dist.png", dpi=120)
    plt.show()


# ─────────────────────────────────────────────
# 3. ONE-HOT ACTIONS (Inchangé, juste n_actions adapté)
# ─────────────────────────────────────────────

def one_hot_actions(A: np.ndarray, n_actions: int) -> np.ndarray:
    """Convertit des actions (N,) → (N, n_actions)."""
    A = np.asarray(A)
    if A.ndim == 2 and A.shape[1] == 1:
        A = A[:, 0]
    
    A = A.astype(np.int64)

    if A.size and (A.min() < 0 or A.max() >= n_actions):
        raise ValueError(f"Actions hors bornes: min={A.min()}, max={A.max()}, n_actions={n_actions}")
    
    return np.eye(n_actions, dtype=np.float32)[A]

# ASTUCE : La fonction normalize() a été supprimée car S et SN 
# ne contiennent que des 0 et des 1 (matrices binaires).


# ─────────────────────────────────────────────
# 4. SPLIT TRAIN / TEST (Inchangé)
# ─────────────────────────────────────────────

def split_dataset(S, A, SN, ratio: float = 0.7, ratio_val: float = 0.10, ratio_test: float = 0.2, shuffle: bool = True, seed: int = 0):
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
# 5. VÉRIFICATION RAPIDE (Adapté pour Precedence)
# ─────────────────────────────────────────────

def quick_check(num_tasks: int):
    """Joue un seul épisode et affiche les 3 premières transitions."""
    env = PrecedenceEnv(num_tasks=num_tasks)
    rng = np.random.default_rng(42)
    obs, _ = env.reset(seed=42)
    
    print("\n--- Vérification : 3 premières transitions ---")
    
    for step in range(3):
        # On utilise notre politique mixte pour ne choisir que des actions valides
        action = get_mixed_action(obs, rng, epsilon=1.0) # 1.0 = exploration 100% (mais légale)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        
        # On affiche le statut des tâches pour voir l'évolution
        s_comp = obs["completed"]
        sn_comp = next_obs["completed"]
        
        print(f"Step {step+1}:")
        print(f"  État t     (Completed) : {s_comp}")
        print(f"  Action     (Tâche)     : {action}")
        print(f"  État t+1   (Completed) : {sn_comp}")
        print(f"  Récompense             : {reward}\n")
        
        if terminated:
            print("  -> Projet terminé !")
            break
            
        obs = next_obs
    env.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Vérifier que la logique fonctionne
    quick_check(num_tasks=NUM_TASKS)

    # 2. Lancer la collecte (On utilise la nouvelle fonction)
    S, A, SN = collect_data_precedence(num_tasks=NUM_TASKS, n_steps=100_000, seed=0)
    
    # 3. Analyser
    analyze_dataset(S, A, SN, num_tasks=NUM_TASKS)

    # 4. Conversion One-Hot pour les actions
    A_oh = one_hot_actions(A, n_actions=NUM_TASKS)

    # 5. Split (SANS normalisation derrière)
    s_tr, a_tr, sn_tr, s_val, a_val, sn_val, s_te, a_te, sn_te = split_dataset(S, A_oh, SN, seed=0)

    # 6. Sauvegarde propre
    np.savez("precedence_data.npz",
         s_train=s_tr, a_train=a_tr, sn_train=sn_tr,
         s_val=s_val,  a_val=a_val,  sn_val=sn_val,
         s_test=s_te,  a_test=a_te,  sn_test=sn_te)
         
    print("\nFichier sauvegardé : precedence_data.npz")
    print("Prêt pour la création du DNN PyTorch ! 🚀")