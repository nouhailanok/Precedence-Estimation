"""Génère les 5 notebooks d'ablation Phase 5."""
import json
from pathlib import Path

OUT = Path(__file__).parent

SETUP = """import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd().parent))

import matplotlib.pyplot as plt
import numpy as np
from ablation_utils import *
"""

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}

def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [text]}

def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

notebooks = {
    "ablation_with_vs_without.ipynb": [
        md("# Ablation — Axe 1 : With vs Without Precedence\n\n**Question :** Le precedence network améliore-t-il l'apprentissage ?\n\nCompare baseline vs precedence (meilleure config) pour DQN et PPO sur CartPole et CliffWalking."),
        code(SETUP),
        code("""# False = charge les résultats déjà entraînés (rapide)
# True  = ré-entraîne DQN CartPole baseline + precedence (3 seeds, ~300 ep)
RUN_TRAINING = False
ENVS = ["CartPole-v1", "CliffWalking-v1"]
ALGOS = ["dqn", "ppo"]
"""),
        code("""results = {"entries": []}

for env in ENVS:
    for algo in ALGOS:
        base = load_baseline_metrics(algo, env)
        if base:
            base["env_name"] = env
            results["entries"].append(base)
            print(f"✓ baseline {algo} {env}: final={base['train_final_mean']:.1f} eval={base['eval_mean']:.1f}")
        else:
            print(f"⚠ baseline manquant: {algo} {env}")

        best_cfg = pick_best_wm_config(algo, env)
        prec = load_precedence_config(algo, env, best_cfg)
        if prec:
            prec["label"] = f"{algo.upper()}+prec (config{best_cfg})"
            results["entries"].append(prec)
            print(f"✓ precedence {algo} {env} cfg{best_cfg}: final={prec['train_final_mean']:.1f}")
        else:
            print(f"⚠ precedence manquant: {algo} {env}")

if RUN_TRAINING and "CartPole-v1" in ENVS:
    print("\\nEntraînement DQN CartPole (baseline + precedence best)...")
    base_runs = run_multi_seed_cartpole(
        lambda seed, **kw: train_dqn_cartpole(seed=seed, use_precedence=False, **kw))
    prec_runs = run_multi_seed_cartpole(
        lambda seed, **kw: train_dqn_cartpole(seed=seed, use_precedence=True, wm_config=2, **kw))
    results["dqn_cartpole_baseline_trained"] = base_runs
    results["dqn_cartpole_precedence_trained"] = prec_runs
"""),
        code("""# Bar chart — final mean (50 derniers épisodes) + eval greedy
plot_groups = {}
for e in results["entries"]:
    key = f"{e['env_name']}\\n{e['label']}"
    plot_groups[key] = {"rewards": e.get("rewards", []), "train_final_mean": e["train_final_mean"]}

plot_bar_comparison(
    results["entries"],
    title="Axe 1 — Baseline vs Precedence (final mean ± std)",
    save_name="ablation_with_vs_without_bars.png",
)

for env in ENVS:
    subset = [e for e in results["entries"] if e["env_name"] == env]
    curves = {e["label"]: {"rewards": e.get("rewards", []), "reward_curves": [e.get("rewards", [])]} for e in subset if e.get("rewards")}
    if curves:
        plot_learning_curves(curves, f"Axe 1 — {env}", f"ablation_with_vs_without_{env.replace('-','_')}.png",
                             solve_threshold=ENV_SPECS[env]["solve_threshold"])

save_ablation_result("ablation_with_vs_without", results)
"""),
    ],

    "ablation_world_model_config.ipynb": [
        md("# Ablation — Axe 2 : World Model Config\n\n**Question :** Quelle config WM (1=S(t+1), 2=S(t+n), 3=ΔS) est la plus utile ?\n\nMulti-seeds optionnel pour barres d'erreur."),
        code(SETUP),
        code("""RUN_TRAINING = False   # True → entraîne DQN+prec configs 1/2/3 (CartPole, 3 seeds)
ENVS = ["CartPole-v1", "CliffWalking-v1"]
CONFIGS = [1, 2, 3]
"""),
        code("""results = {}

for env in ENVS:
    results[env] = {}
    for cfg in CONFIGS:
        entry = load_precedence_config("dqn", env, cfg)
        if entry:
            entry["label"] = f"Config {cfg}"
            results[env][f"config_{cfg}"] = entry
            print(f"{env} config{cfg}: final={entry['train_final_mean']:.1f} eval={entry.get('eval_mean',0):.1f}")
        else:
            print(f"⚠ {env} config{cfg} — pas de JSON")

if RUN_TRAINING:
    results["CartPole-v1_trained"] = {}
    for cfg in CONFIGS:
        results["CartPole-v1_trained"][f"config_{cfg}"] = run_multi_seed_cartpole(
            train_dqn_cartpole, wm_config=cfg, use_precedence=True)
        print(f"trained config{cfg}: {results['CartPole-v1_trained'][f'config_{cfg}']['final_mean']:.1f}")
"""),
        code("""for env in ENVS:
    entries = list(results.get(env, {}).values())
    if entries:
        plot_bar_comparison(entries, f"Axe 2 — WM Config — {env}",
                            f"ablation_wm_config_{env.replace('-','_')}.png")
        curves = {e["label"]: {"reward_curves": [e.get("rewards", [])]} for e in entries if e.get("rewards")}
        if curves:
            plot_learning_curves(curves, f"WM Config curves — {env}",
                                 f"ablation_wm_config_curves_{env.replace('-','_')}.png",
                                 solve_threshold=ENV_SPECS[env]["solve_threshold"])

save_ablation_result("ablation_world_model_config", results)
"""),
    ],

    "ablation_verif_strategy.ipynb": [
        md("# Ablation — Axe 3 : Stratégie de vérification\n\n**Question :** Quel critère de vérification est le plus utile ?\n\n- **CartPole** : `safety` | `stability` | `combined`\n- **CliffWalking** : `safety` | `lookahead` | `combined` (notebook CliffWalking)\n\n`RUN_TRAINING=True` entraîne CartPole config 1 avec les 3 stratégies."),
        code(SETUP),
        code("""RUN_TRAINING = False
WM_CONFIG = 1  # fixer config WM pour isoler l'effet stratégie
"""),
        code("""results = {}

# CartPole — entraînement multi-stratégie
if RUN_TRAINING:
    for strat in ENV_SPECS["CartPole-v1"]["verif_strategies"]:
        results[f"cartpole_{strat}"] = run_multi_seed_cartpole(
            train_dqn_cartpole, use_precedence=True, wm_config=WM_CONFIG, verify_strategy=strat)
        print(f"CartPole {strat}: {results[f'cartpole_{strat}']['final_mean']:.1f}")
else:
    # Charge config 1 comme proxy (stratégie combined par défaut dans les JSON)
    e = load_precedence_config("dqn", "CartPole-v1", WM_CONFIG)
    if e:
        e["label"] = "combined (saved)"
        results["cartpole_combined_saved"] = e
    print("⚠ Pour safety/stability séparés: RUN_TRAINING=True ou entraîner manuellement")

# CliffWalking — charge résultats existants (combined dans JSON)
for cfg in [1, 2, 3]:
    e = load_precedence_config("dqn", "CliffWalking-v1", cfg)
    if e:
        results[f"cliff_config{cfg}"] = e
"""),
        code("""entries = []
for k, v in results.items():
    if isinstance(v, dict) and "final_mean" in v:
        entries.append({"label": k, "train_final_mean": v["final_mean"], "train_final_std": v["final_std"],
                        "eval_mean": v.get("eval_mean", 0), "rewards": v["reward_curves"][0] if v.get("reward_curves") else []})
    elif isinstance(v, dict) and "train_final_mean" in v:
        entries.append(v)

if entries:
    plot_bar_comparison(entries, "Axe 3 — Verification Strategy", "ablation_verif_strategy.png")

save_ablation_result("ablation_verif_strategy", results)
"""),
    ],

    "ablation_algo.ipynb": [
        md("# Ablation — Axe 4 : DQN vs PPO\n\n**Question :** Le gain du precedence network dépend-il de l'algorithme RL ?\n\nCompare DQN+prec vs PPO+prec (meilleure config WM) sur CartPole et CliffWalking."),
        code(SETUP),
        code("""ENVS = ["CartPole-v1", "CliffWalking-v1"]
results = {}
"""),
        code("""for env in ENVS:
    results[env] = {}
    for algo in ["dqn", "ppo"]:
        cfg = pick_best_wm_config(algo, env)
        entry = load_precedence_config(algo, env, cfg)
        if entry:
            entry["label"] = f"{algo.upper()}+prec cfg{cfg}"
            results[env][algo] = entry
            print(f"{env} {algo}+prec cfg{cfg}: final={entry['train_final_mean']:.1f} eval={entry.get('eval_mean',0):.1f}")
"""),
        code("""for env in ENVS:
    entries = list(results.get(env, {}).values())
    if len(entries) >= 2:
        plot_bar_comparison(entries, f"Axe 4 — DQN vs PPO + precedence — {env}",
                            f"ablation_algo_{env.replace('-','_')}.png")
        curves = {e["label"]: {"reward_curves": [e.get("rewards", [])]} for e in entries if e.get("rewards")}
        if curves:
            plot_learning_curves(curves, f"DQN vs PPO — {env}", f"ablation_algo_curves_{env.replace('-','_')}.png",
                                 solve_threshold=ENV_SPECS[env]["solve_threshold"])

save_ablation_result("ablation_algo", results)
"""),
    ],

    "ablation_summary.ipynb": [
        md("# Ablation — Résumé global\n\nCharge tous les JSON de `results/` et produit table + plots comparatifs."),
        code(SETUP),
        code("""all_res = load_all_ablation_results()
print(f"Fichiers trouvés: {list(all_res.keys())}")
"""),
        code("""table = make_summary_table(all_res)
print(table)
"""),
        code("""# Bar chart global — toutes les expériences avec train_final_mean
entries = []
for name, data in all_res.items():
    if name == "ablation_with_vs_without" and "entries" in data:
        for e in data["entries"]:
            e2 = dict(e)
            e2["label"] = f"{name[:12]}\\n{e.get('label','')[:20]}"
            entries.append(e2)
    elif isinstance(data, dict):
        for subk, subv in data.items():
            if isinstance(subv, dict) and "train_final_mean" in subv:
                entries.append({**subv, "label": f"{name}\\n{subk}"})
            elif isinstance(subv, dict) and "final_mean" in subv:
                entries.append({"label": f"{name}\\n{subk}", "train_final_mean": subv["final_mean"],
                                "train_final_std": subv["final_std"], "eval_mean": subv.get("eval_mean", 0)})

if entries:
    plot_bar_comparison(entries[:12], "Ablation Summary — all experiments", "ablation_summary_all.png")
"""),
        code("""# Superposer courbes principales (with vs without)
if "ablation_with_vs_without" in all_res:
    data = all_res["ablation_with_vs_without"]
    for env in ["CartPole-v1", "CliffWalking-v1"]:
        subset = [e for e in data.get("entries", []) if e.get("env_name") == env and e.get("rewards")]
        if subset:
            curves = {e["label"]: {"reward_curves": [e["rewards"]]} for e in subset}
            plot_learning_curves(curves, f"Summary — {env}", f"ablation_summary_curves_{env.replace('-','_')}.png",
                                 solve_threshold=ENV_SPECS[env]["solve_threshold"])
"""),
    ],
}

for fname, cells in notebooks.items():
    path = OUT / fname
    with open(path, "w") as f:
        json.dump(nb(cells), f, indent=1)
    print(f"✅ {path}")
