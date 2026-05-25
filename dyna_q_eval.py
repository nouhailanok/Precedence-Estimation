"""
dyna_q_eval.py — Évaluation d'un agent DQN/Dyna‑Q entraîné sur CartPole
- Évalue dynaq_agent_X.pth sur N_EPISODES independants (greedy)
- Affiche score moyen, std, min, max
- Histogramme des scores sauvegardé
- Prêt à être utilisé pour rapport, ablation, etc.

Usage :
    python dyna_q_eval.py --checkpoint dynaq_agent_10.pth --episodes 50
    python dyna_q_eval.py --checkpoint CREATEAGENT_plots/dynaq_agent_cl_correction_10.pth --episodes 200
    

"""

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import argparse
import matplotlib.pyplot as plt
import os
import random

# === CONFIG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_EVAL_AG_DIR = os.path.join(BASE_DIR, "EVALDyna_Q_plots")
os.makedirs(PLOTS_EVAL_AG_DIR, exist_ok=True)



# ==== AGENT ARCHI ====
class QNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, x):
        return self.net(x)

def evaluate_agent(checkpoint_path, env_name="CartPole-v1", n_episodes=50, seed_base=123, eps_eval=0.05):
    # -- Load Agent --
    torch.manual_seed(seed_base)
    np.random.seed(seed_base)
    random.seed(seed_base)
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    n_act   = env.action_space.n

    agent = QNet(obs_dim, n_act)
    agent.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    agent.eval()

    print(f"Évaluation du modèle clean : {checkpoint_path}")

    scores = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_base+ep)
        done = False
        total = 0
        while not done:
            # 🔥 stochastic policy (important)
            if np.random.rand() < eps_eval:
                act = env.action_space.sample()   # exploration même en test
            else:
                with torch.no_grad():
                    qvals = agent(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                    act = int(torch.argmax(qvals).item())
            obs, reward, terminated, truncated, _ = env.step(act)
            total += reward
            done = terminated or truncated
        scores.append(total)
    env.close()
    scores = np.array(scores)
    print(f"\nRésultats sur {n_episodes} épisodes greedy clean :")
    print(f"  Moyenne : {np.mean(scores):.2f}")
    print(f"  Écart-type : {np.std(scores):.2f}")
    print(f"  Min / Max : {np.min(scores)} / {np.max(scores)}")

    # -- Histogramme
    plt.figure(figsize=(8,4))
    plt.hist(scores, bins=15, color="#378ADD", edgecolor="black", alpha=0.83)
    plt.xlabel("Score (reward épisode)")
    plt.ylabel("Nombre d’épisodes")
    plt.title(f"Distribution scores agent ({os.path.basename(checkpoint_path)})")
    plt.grid(alpha=0.3)
    fname = f"scores_normal_{os.path.splitext(os.path.basename(checkpoint_path))[0]}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_EVAL_AG_DIR, fname), dpi=120)
    plt.show()
    print(f"Figure sauvegardée → {fname}")

    # -- Tableau pour le rapport (optionnel)
    with open(f"eval_{os.path.splitext(os.path.basename(checkpoint_path))[0]}.txt", "w") as f:
        f.write(f"Checkpoint : {checkpoint_path}\n")
        f.write(f"Épisodes : {n_episodes}\n")
        f.write(f"Moyenne : {np.mean(scores):.2f}\n")
        f.write(f"Std : {np.std(scores):.2f}\n")
        f.write(f"Min : {np.min(scores)}\n")
        f.write(f"Max : {np.max(scores)}\n")
        f.write(f"Scores bruts : {', '.join(map(str, scores))}\n")
    print(f"Résumé texte sauvegardé → eval_{os.path.splitext(os.path.basename(checkpoint_path))[0]}.txt")

    return scores

def evaluate_agent_noisy(checkpoint_path, env_name="CartPole-v1", n_episodes=50, seed_base=123,noise_std=0.01):
    # -- Load Agent --
    torch.manual_seed(seed_base)
    np.random.seed(seed_base)
    random.seed(seed_base)
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    n_act   = env.action_space.n

    agent = QNet(obs_dim, n_act)
    agent.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    agent.eval()

    print(f"Évaluation du modèle  avec noise : {checkpoint_path}")

    scores = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_base+ep)
        done = False
        total = 0
        while not done:
            # ajouter bruit sur observation AVANT réseau
            obs_in = obs + np.random.normal(0, noise_std, size=len(obs))

            with torch.no_grad():
                qvals = agent(torch.tensor(obs_in, dtype=torch.float32).unsqueeze(0))
                act = int(torch.argmax(qvals).item())

            obs, reward, terminated, truncated, _ = env.step(act)
            total += reward
            done = terminated or truncated
        scores.append(total)
    env.close()
    scores = np.array(scores)
    print(f"\nRésultats sur {n_episodes} épisodes greedy noisy :")
    print(f"  Moyenne : {np.mean(scores):.2f}")
    print(f"  Écart-type : {np.std(scores):.2f}")
    print(f"  Min / Max : {np.min(scores)} / {np.max(scores)}")

    # -- Histogramme
    plt.figure(figsize=(8,4))
    plt.hist(scores, bins=15, color="#378ADD", edgecolor="black", alpha=0.83)
    plt.xlabel("Score (reward épisode)")
    plt.ylabel("Nombre d’épisodes")
    plt.title(f"Distribution scores agent ({os.path.basename(checkpoint_path)})")
    plt.grid(alpha=0.3)
    fname = f"scores_noisy_{os.path.splitext(os.path.basename(checkpoint_path))[0]}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_EVAL_AG_DIR, fname), dpi=120)
    plt.show()
    print(f"Figure sauvegardée → {fname}")

    # -- Tableau pour le rapport (optionnel)
    with open(f"eval_{os.path.splitext(os.path.basename(checkpoint_path))[0]}.txt", "w") as f:
        f.write(f"Checkpoint : {checkpoint_path}\n")
        f.write(f"Épisodes : {n_episodes}\n")
        f.write(f"Moyenne : {np.mean(scores):.2f}\n")
        f.write(f"Std : {np.std(scores):.2f}\n")
        f.write(f"Min : {np.min(scores)}\n")
        f.write(f"Max : {np.max(scores)}\n")
        f.write(f"Scores bruts : {', '.join(map(str, scores))}\n")
    print(f"Résumé texte sauvegardé → eval_{os.path.splitext(os.path.basename(checkpoint_path))[0]}.txt")

    return scores

def print_stats(name, scores):
    print(f"\n{name}")
    print(f"Mean: {np.mean(scores):.2f}")
    print(f"Std: {np.std(scores):.2f}")
    print(f"Min: {np.min(scores)}")
    print(f"Max: {np.max(scores)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help=".pth à évaluer")
    parser.add_argument("--episodes", type=int, default=50, help="nb épisodes de test")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Nom env gymnasium")
    args = parser.parse_args()
    evaluate_agent(args.checkpoint, env_name=args.env, n_episodes=args.episodes)
    scores_clean = evaluate_agent(args.checkpoint, env_name=args.env, n_episodes=args.episodes)
    scores_noisy = evaluate_agent_noisy(args.checkpoint, env_name=args.env, n_episodes=args.episodes, noise_std=0.02)
    print_stats("Clean evaluation", scores_clean)
    print_stats("Noisy evaluation", scores_noisy)