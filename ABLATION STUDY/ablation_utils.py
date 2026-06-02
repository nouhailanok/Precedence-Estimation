"""
ablation_utils.py — Utilitaires partagés pour l'étude d'ablation (Phase 5).

Axes couverts :
  1. with vs without precedence
  2. world model config 1 / 2 / 3
  3. stratégie de vérification (CartPole: safety/stability/combined)
  4. algorithme DQN vs PPO

Usage depuis les notebooks :
    import sys
    sys.path.insert(0, "..")  # ou chemin projet
    from ablation_utils import *
"""

from __future__ import annotations

import json
import os
import random
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ── Chemins ──────────────────────────────────────────────────────────
ABLATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ABLATION_DIR.parent
RESULTS_DIR = ABLATION_DIR / "results"
PLOTS_DIR = ABLATION_DIR / "plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Protocole eval aligné dqn_with_precedence_CartPole_full.ipynb / dqn_agent.py
N_SEEDS_DEFAULT = 3
SEEDS_DEFAULT = [42, 123, 456]
N_EPISODES_ABLATION = 300
EVAL_N_EPISODES = 30
FINAL_WINDOW = 50

ENV_SPECS = {
    "CartPole-v1": {
        "solve_threshold": 500.0,
        "data_path": PROJECT_ROOT / "data" / "collected" / "CartPole",
        "wm_dir": PROJECT_ROOT / "world_model" / "checkpoints_CartPole",
        "state_dim": 4,
        "action_dim": 2,
        "wm_n_step": 5,
        "max_steps": 500,
        "discrete_obs": False,
        "verif_strategies": ("safety", "stability", "combined"),
        "dqn_baseline_metrics": PROJECT_ROOT / "DQN_plots" / "CartPole-v1" / "dqn_metrics.json",
        "ppo_baseline_metrics": PROJECT_ROOT / "PPO_plots" / "CartPole-v1" / "ppo_metrics.json",
        "dqn_prec_metrics": PROJECT_ROOT / "world_model" / "checkpoints_dqn_with_precedence_CartPole" / "dqn_with_precedence_CartPole_metrics.json",
        "dqn_prec_legacy": PROJECT_ROOT / "world_model" / "checkpoints_with_precedence" / "dqn_with_precedence_results.json",
        "ppo_prec_metrics": PROJECT_ROOT / "world_model" / "checkpoints_ppo_with_precedence" / "ppo_with_precedence_results.json",
    },
    "CliffWalking-v1": {
        "solve_threshold": -20.0,
        "data_path": PROJECT_ROOT / "data" / "collected" / "CliffWalking",
        "wm_dir": PROJECT_ROOT / "world_model" / "checkpoints_CliffWalking",
        "state_dim": 48,
        "action_dim": 4,
        "wm_n_step": 2,
        "max_steps": 200,
        "discrete_obs": True,
        "n_states": 48,
        # CliffWalking notebook utilise lookahead au lieu de stability
        "verif_strategies": ("safety", "lookahead", "combined"),
        "dqn_baseline_metrics": PROJECT_ROOT / "DQN_plots" / "CliffWalking-v1" / "dqn_metrics.json",
        "ppo_baseline_metrics": PROJECT_ROOT / "PPO_plots" / "CliffWalking-v1" / "ppo_metrics.json",
        "dqn_prec_metrics": None,
        "dqn_prec_legacy": PROJECT_ROOT / "world_model" / "checkpoints_dqn_precedence_CliffWalking" / "dqn_precedence_cliffwalking_results.json",
        "ppo_prec_metrics": PROJECT_ROOT / "world_model" / "checkpoints_ppo_precedence_CliffWalking" / "ppo_precedence_cliffwalking_results.json",
    },
}


# ── Métriques ────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def learning_curve_auc(rewards: list | np.ndarray) -> float:
    arr = np.asarray(rewards, dtype=np.float64)
    if len(arr) == 0:
        return 0.0
    if len(arr) == 1:
        return float(arr[0])
    return float(np.trapz(arr, np.arange(len(arr))))


def summarize_rewards(rewards: list, n_steps: int = 0, eval_block: dict | None = None) -> dict:
    arr = np.asarray(rewards, dtype=np.float32)
    last_n = min(FINAL_WINDOW, len(arr))
    out = {
        "rewards": [float(r) for r in rewards],
        "train_avg_reward": float(arr.mean()) if len(arr) else 0.0,
        "train_std_reward": float(arr.std()) if len(arr) else 0.0,
        "train_final_mean": float(arr[-last_n:].mean()) if last_n else 0.0,
        "train_final_std": float(arr[-last_n:].std()) if last_n > 1 else 0.0,
        "train_auc": learning_curve_auc(arr),
        "n_episodes": len(rewards),
        "n_steps": int(n_steps),
    }
    if eval_block:
        out.update(eval_block)
    return out


def aggregate_seeds(runs: list[dict]) -> dict:
    """Agrège plusieurs seeds → mean ± std sur final_mean et eval_mean."""
    finals = [r["train_final_mean"] for r in runs]
    evals = [r.get("eval_mean", r["train_final_mean"]) for r in runs]
    aucs = [r["train_auc"] for r in runs]
    return {
        "seeds": [r.get("seed") for r in runs],
        "runs": runs,
        "final_mean": float(np.mean(finals)),
        "final_std": float(np.std(finals)),
        "eval_mean": float(np.mean(evals)),
        "eval_std": float(np.std(evals)),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "reward_curves": [r["rewards"] for r in runs],
    }


# ── Chargement résultats existants ───────────────────────────────────
def _load_json(path: Path) -> dict | None:
    if path and path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_baseline_metrics(algo: str, env_name: str) -> dict | None:
    spec = ENV_SPECS[env_name]
    path = spec[f"{algo}_baseline_metrics"]
    data = _load_json(path)
    if not data:
        return None
    rewards = data.get("train", {})
    # dqn_metrics.json n'a pas rewards list — reconstruire depuis eval si absent
    entry = {
        "label": f"{algo.upper()} baseline",
        "algo": algo,
        "env_name": env_name,
        "use_precedence": False,
        "train_final_mean": data["train"].get("train_final_mean", 0),
        "train_final_std": data["train"].get("train_final_std", 0),
        "train_avg_reward": data["train"].get("train_avg_reward", 0),
        "train_std_reward": data["train"].get("train_std_reward", 0),
        "train_auc": data["train"].get("train_auc", 0),
        "n_episodes": data["train"].get("n_episodes", 0),
        "n_steps": data["train"].get("n_steps", 0),
        "eval_mean": data["eval"].get("eval_mean", 0),
        "eval_std": data["eval"].get("eval_std", 0),
        "success_rate": data["eval"].get("success_rate", 0),
        "source": str(path),
    }
    return entry


def load_precedence_config(algo: str, env_name: str, config: int) -> dict | None:
    spec = ENV_SPECS[env_name]
    key = f"config_{config}"

    if algo == "dqn":
        data = _load_json(spec.get("dqn_prec_metrics")) or _load_json(spec.get("dqn_prec_legacy"))
    else:
        data = _load_json(spec.get("ppo_prec_metrics"))

    if not data or key not in data:
        return None

    block = data[key]
    rewards = block.get("rewards", [])
    last_n = min(FINAL_WINDOW, len(rewards))
    return {
        "label": f"{algo.upper()}+prec config{config}",
        "algo": algo,
        "env_name": env_name,
        "use_precedence": True,
        "wm_config": config,
        "verify_strategy": block.get("verify_strategy", "combined"),
        "rewards": rewards,
        "train_final_mean": block.get("final_reward_mean", float(np.mean(rewards[-last_n:])) if last_n else 0),
        "train_final_std": block.get("final_reward_std", float(np.std(rewards[-last_n:])) if last_n > 1 else 0),
        "train_avg_reward": float(np.mean(rewards)) if rewards else 0,
        "train_std_reward": float(np.std(rewards)) if rewards else 0,
        "train_auc": learning_curve_auc(rewards),
        "n_episodes": len(rewards),
        "eval_mean": block.get("eval_mean", 0),
        "eval_std": block.get("eval_std", 0),
        "success_rate": block.get("success_rate", 0),
        "source": str(spec.get("dqn_prec_metrics") or spec.get("ppo_prec_metrics") or spec.get("dqn_prec_legacy")),
    }


def pick_best_wm_config(algo: str, env_name: str) -> int:
    best_cfg, best_score = 1, -np.inf
    for cfg in (1, 2, 3):
        entry = load_precedence_config(algo, env_name, cfg)
        if entry and entry["train_final_mean"] > best_score:
            best_score = entry["train_final_mean"]
            best_cfg = cfg
    return best_cfg


def save_ablation_result(name: str, payload: dict) -> Path:
    path = RESULTS_DIR / f"{name}.json"
    payload["saved_at"] = datetime.now().isoformat()
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"✅ Saved → {path}")
    return path


def load_all_ablation_results() -> dict[str, dict]:
    out = {}
    for p in sorted(RESULTS_DIR.glob("*.json")):
        with open(p) as f:
            out[p.stem] = json.load(f)
    return out


# ── CartPole DQN + precedence (entraînement ablation) ───────────────
class WorldModel(nn.Module):
    def __init__(self, obs_dim, act_dim, config=1, n_step=5, hidden=64):
        super().__init__()
        self.config = config
        self.n_step = n_step
        in_dim = obs_dim + (n_step * act_dim if config == 2 else act_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, obs_dim),
        )

    def predict(self, state, action_input):
        out = self.net(torch.cat([state, action_input], dim=-1))
        return state + out if self.config == 3 else out


class CartPoleActionVerifier:
    def __init__(self, world_model, config, aux_model=None, agent_qnet=None,
                 safety_threshold=0.3, stability_threshold=0.5):
        self.model = world_model
        self.config = config
        self.aux_model = aux_model
        self.agent_qnet = agent_qnet
        self.safety_threshold = safety_threshold
        self.stability_threshold = stability_threshold
        self.pole_angle_safe_bound = 0.2
        self.cart_position_safe_bound = 2.0

    @torch.no_grad()
    def _build_action_sequence_config2(self, state_t, first_action, action_dim, device):
        n_step = self.model.n_step
        actions_seq, state_loop = [], state_t.clone()
        for step in range(n_step):
            if step == 0:
                a = first_action
            elif self.agent_qnet is not None:
                a = self.agent_qnet(state_loop).argmax(1).item()
            else:
                a = first_action
            a_oh = torch.zeros(action_dim, device=device)
            a_oh[a] = 1.0
            actions_seq.append(a_oh)
            if step < n_step - 1 and self.aux_model is not None:
                state_loop = self.aux_model.predict(state_loop, a_oh.unsqueeze(0))
        return torch.cat(actions_seq).unsqueeze(0)

    @torch.no_grad()
    def predict_next_state(self, state_np, action, action_dim, device):
        state_t = torch.FloatTensor(state_np).unsqueeze(0).to(device)
        if self.config == 2:
            action_input = self._build_action_sequence_config2(state_t, action, action_dim, device)
        else:
            action_input = torch.zeros(1, action_dim, device=device)
            action_input[0, action] = 1.0
        return self.model.predict(state_t, action_input).cpu().numpy()[0]

    @torch.no_grad()
    def verify(self, state_np, action, action_dim, device, strategy="combined") -> bool:
        pred = self.predict_next_state(state_np, action, action_dim, device)
        pole_angle, cart_pos = abs(pred[2]), abs(pred[0])
        safety_ok = pole_angle < self.pole_angle_safe_bound and cart_pos < self.cart_position_safe_bound
        if strategy in ("safety", "combined") and not safety_ok:
            return False
        if strategy in ("stability", "combined"):
            preds = [
                self.predict_next_state(state_np + np.random.randn(len(state_np)) * 0.01, action, action_dim, device)
                for _ in range(5)
            ]
            var = float(np.mean(np.var(np.array(preds), axis=0)))
            if strategy == "stability":
                return var < self.stability_threshold
            if strategy == "combined" and var >= self.stability_threshold:
                return False
        return True


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def _load_wm_cartpole(config, spec, device):
    model = WorldModel(spec["state_dim"], spec["action_dim"], config=config,
                       n_step=spec["wm_n_step"]).to(device)
    p = spec["wm_dir"] / f"wm_CartPole-v1_config{config}.pt"
    if config == 2:
        p = spec["wm_dir"] / f"wm_CartPole-v1_config{config}_n5.pt"
    ckpt = torch.load(p, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def evaluate_dqn_greedy(q_net, env_name, state_mean, state_std, spec, device=DEVICE):
    env = gym.make(env_name)
    rewards, lengths = [], []
    q_net.eval()
    for ep in range(EVAL_N_EPISODES):
        state, _ = env.reset(seed=SEEDS_DEFAULT[0] + 1000 + ep)
        s = (state - state_mean) / (state_std + 1e-8)
        ep_r, ep_len = 0.0, 0
        for _ in range(spec["max_steps"]):
            st = torch.FloatTensor(s).unsqueeze(0).to(device)
            a = int(q_net(st).argmax(1).item())
            ns, r, term, trunc, _ = env.step(a)
            s = (ns - state_mean) / (state_std + 1e-8)
            ep_r += r
            ep_len += 1
            if term or trunc:
                break
        rewards.append(ep_r)
        lengths.append(ep_len)
    env.close()
    arr = np.array(rewards, dtype=np.float32)
    thr = spec["solve_threshold"]
    return {
        "eval_mean": float(arr.mean()),
        "eval_std": float(arr.std()),
        "eval_min": float(arr.min()),
        "eval_max": float(arr.max()),
        "eval_median": float(np.median(arr)),
        "eval_length_mean": float(np.mean(lengths)),
        "success_rate": float((arr >= thr).mean()),
        "n_solved": int((arr >= thr).sum()),
        "eval_rewards": [float(x) for x in rewards],
    }


def train_dqn_cartpole(
    seed: int,
    use_precedence: bool = False,
    wm_config: int = 1,
    verify_strategy: str = "combined",
    n_episodes: int = N_EPISODES_ABLATION,
) -> dict:
    """Entraîne DQN CartPole (baseline ou + precedence) pour une seed."""
    set_seed(seed)
    spec = ENV_SPECS["CartPole-v1"]
    env_name = "CartPole-v1"

    scaler = np.load(spec["data_path"] / "scaler.npz", allow_pickle=True)
    state_mean = scaler["state_mean"].astype(np.float32)
    state_std = scaler["state_std"].astype(np.float32)

    # Hyperparams (alignés notebook full)
    batch_size, lr, gamma = 64, 1e-3, 0.99
    eps_start, eps_end, eps_decay = 1.0, 0.01, 0.995
    target_update, buffer_size = 10, 50_000

    env = gym.make(env_name)
    q_net = QNetwork(spec["state_dim"], spec["action_dim"]).to(DEVICE)
    target = QNetwork(spec["state_dim"], spec["action_dim"]).to(DEVICE)
    target.load_state_dict(q_net.state_dict())
    opt = optim.Adam(q_net.parameters(), lr=lr)
    buf = deque(maxlen=buffer_size)
    epsilon = eps_start

    wm, aux, verifier = None, None, None
    if use_precedence:
        wm = _load_wm_cartpole(wm_config, spec, DEVICE)
        if wm_config == 2:
            aux = _load_wm_cartpole(1, spec, DEVICE)
        verifier = CartPoleActionVerifier(wm, wm_config, aux_model=aux, agent_qnet=q_net)

    rewards, n_steps = [], 0

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        s = (state - state_mean) / (state_std + 1e-8)
        ep_r = 0.0

        for _ in range(spec["max_steps"]):
            if use_precedence and verifier is not None:
                if verifier.config == 2:
                    verifier.agent_qnet = q_net
                if random.random() < epsilon:
                    rec = random.randint(0, spec["action_dim"] - 1)
                else:
                    rec = int(q_net(torch.FloatTensor(s).unsqueeze(0).to(DEVICE)).argmax(1).item())
                action = rec
                if not verifier.verify(s, rec, spec["action_dim"], DEVICE, verify_strategy):
                    for alt in range(spec["action_dim"]):
                        if alt != rec and verifier.verify(s, alt, spec["action_dim"], DEVICE, verify_strategy):
                            action = alt
                            break
            else:
                if random.random() < epsilon:
                    action = random.randint(0, spec["action_dim"] - 1)
                else:
                    action = int(q_net(torch.FloatTensor(s).unsqueeze(0).to(DEVICE)).argmax(1).item())

            ns, r, term, trunc, _ = env.step(action)
            ns_n = (ns - state_mean) / (state_std + 1e-8)
            done = term or trunc
            buf.append((s, action, r, ns_n, done))
            s, ep_r, n_steps = ns_n, ep_r + r, n_steps + 1

            if len(buf) >= batch_size:
                batch = random.sample(buf, batch_size)
                states, actions, rews, nexts, dones = zip(*batch)
                st = torch.FloatTensor(np.array(states)).to(DEVICE)
                ac = torch.LongTensor(actions).to(DEVICE)
                rw = torch.FloatTensor(rews).to(DEVICE)
                nx = torch.FloatTensor(np.array(nexts)).to(DEVICE)
                dn = torch.FloatTensor(dones).to(DEVICE)
                qv = q_net(st).gather(1, ac.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    ba = q_net(nx).argmax(1)
                    tq = target(nx).gather(1, ba.unsqueeze(1)).squeeze(1)
                    tgt = rw + gamma * tq * (1 - dn)
                loss = nn.MSELoss()(qv, tgt)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
                opt.step()

            if done:
                break

        rewards.append(ep_r)
        epsilon = max(eps_end, epsilon * eps_decay)
        if (ep + 1) % target_update == 0:
            target.load_state_dict(q_net.state_dict())

    env.close()
    eval_m = evaluate_dqn_greedy(q_net, env_name, state_mean, state_std, spec)
    result = summarize_rewards(rewards, n_steps, eval_m)
    result.update({
        "seed": seed,
        "use_precedence": use_precedence,
        "wm_config": wm_config if use_precedence else None,
        "verify_strategy": verify_strategy if use_precedence else None,
        "algo": "dqn",
        "env_name": env_name,
    })
    return result


def run_multi_seed_cartpole(train_fn, seeds: list[int] | None = None, **kwargs) -> dict:
    seeds = seeds or SEEDS_DEFAULT
    runs = [train_fn(seed=seed, **kwargs) for seed in seeds]
    return aggregate_seeds(runs)


# ── Plotting ─────────────────────────────────────────────────────────
def plot_learning_curves(groups: dict[str, dict], title: str, save_name: str,
                         solve_threshold: float | None = None):
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, g in groups.items():
        curves = g.get("reward_curves") or [g.get("rewards", [])]
        if not curves or not curves[0]:
            continue
        max_len = max(len(c) for c in curves)
        mat = np.full((len(curves), max_len), np.nan)
        for i, c in enumerate(curves):
            mat[i, : len(c)] = c
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
        x = np.arange(len(mean))
        ax.plot(x, mean, label=label)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15)
    if solve_threshold is not None:
        ax.axhline(solve_threshold, color="r", ls="--", alpha=0.4)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = PLOTS_DIR / save_name
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"📊 {path}")
    return path


def plot_bar_comparison(entries: list[dict], title: str, save_name: str,
                        metric: str = "train_final_mean"):
    labels = [e.get("label", "?") for e in entries]
    means = [e.get(metric, 0) for e in entries]
    stds = [e.get("train_final_std", e.get("eval_std", 0)) for e in entries]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color=plt.cm.tab10(np.linspace(0, 0.8, len(labels))))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = PLOTS_DIR / save_name
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"📊 {path}")
    return path


def make_summary_table(all_results: dict[str, dict]) -> str:
    lines = [f"{'Experiment':<40} {'Final mean±std':>18} {'Eval mean':>12} {'AUC':>10} {'Eps':>6}"]
    lines.append("-" * 90)
    for name, data in sorted(all_results.items()):
        if "final_mean" in data:
            fm, fs = data["final_mean"], data["final_std"]
            em = data.get("eval_mean", 0)
            auc = data.get("auc_mean", 0)
            ne = len(data.get("runs", [{}])[0].get("rewards", []))
        elif "entries" in data:
            continue
        else:
            fm = data.get("train_final_mean", 0)
            fs = data.get("train_final_std", 0)
            em = data.get("eval_mean", 0)
            auc = data.get("train_auc", 0)
            ne = data.get("n_episodes", 0)
        lines.append(f"{name:<40} {fm:>8.1f}±{fs:<8.1f} {em:>12.1f} {auc:>10.0f} {ne:>6}")
    return "\n".join(lines)
