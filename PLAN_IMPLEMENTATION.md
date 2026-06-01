# 🎯 Plan d'Implémentation - Precedence Network

## Contribution Principale
**Comparer RL (DQN/PPO) AVEC vs SANS Precedence Network**

---

## 1️⃣ PRECEDENCE NETWORK (3 Configurations)

### **Config 1: St → St+1**
- Input: `[St | St+1]` (deux états consécutifs)
- Output: Score ∈ [0,1] "Cette transition est-elle valide?"
- Entraînement: Positif = vraies transitions, Négatif = paires d'états aléatoires

### **Config 2: St → St+n (n=5)**
- Input: `[St | St+5]` (états à n steps d'intervalle)
- Output: Score ∈ [0,1] "Cette séquence multi-step est-elle cohérente?"
- Entraînement: Sur sequences_n5.npz

### **Config 3: Delta ΔS = St+1 - St**
- Input: `[St | ΔS]` (état + changement)
- Output: Score ∈ [0,1] "Ce delta est-il physiquement plausible?"
- Entraînement: Sur transitions réelles vs deltas aléatoires

---

## 2️⃣ WORKFLOW AVEC PRECEDENCE NETWORK

```
┌─────────────────────────────────────────────────────────┐
│           RL AGENT AVEC PRECEDENCE NETWORK              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  State St                                               │
│    ↓                                                    │
│  [DQN/PPO] → Action At (candidate)                      │
│    ↓                                                    │
│  [World Model] → Prédire S̃t+1 (without executing)      │
│    ↓                                                    │
│  [Precedence Net] → Score(St, S̃t+1) ∈ [0,1]           │
│    ↓                                                    │
│  ┌─────────────────────────────────────┐               │
│  │ Score > Threshold?                  │               │
│  ├─────────────────────────────────────┤               │
│  │ YES → Execute At in real environment│               │
│  │ NO  → Try alternative action        │               │
│  └─────────────────────────────────────┘               │
│    ↓                                                    │
│  Get real St+1 from environment                        │
│    ↓                                                    │
│  Store (St, At, Rt, St+1) in replay buffer             │
│    ↓                                                    │
│  Update DQN/PPO network                                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 3️⃣ BASELINE (SANS PRECEDENCE NETWORK)

```
State St
  ↓
[DQN/PPO] → Action At
  ↓
Execute At directly in environment
  ↓
Get St+1, Rt
  ↓
Update network
```

---

## 4️⃣ ENVIRONNEMENTS À TESTER

✅ **CartPole-v1** (continu)
✅ **CliffWalking-v1** (discret)
✅ **MountainCar-v0** (continu, exploration difficile)
✅ **LunarLander-v2** (continu, complex)
✅ **Acrobot-v1** (continu, control difficile)

---

## 5️⃣ STRUCTURE DES FICHIERS

```
precedence_network/
├── precedence_network.py          # Architecture PN (3 configs)
├── train_precedence.py             # Script d'entraînement PN
├── precedence_dataset.py           # Création datasets positif/négatif
│
rl_agents/
├── dqn_agent.py                   # DQN baseline + with_precedence
├── ppo_agent.py                   # PPO baseline + with_precedence
├── train_rl.py                    # Script d'entraînement RL
│
experiments/
├── experiment_runner.py           # Orchestrer tous les expériments
├── config.yaml                    # Config des expériments
│
evaluation/
├── metrics.py                     # Calcul des métriques
├── plot_results.py                # Visualisations
├── compare_methods.py             # Comparaison WITH vs WITHOUT
```

---

## 6️⃣ PIPELINE D'EXPÉRIENCE

### **Phase 1: Entraîner les World Models** ✅ (DÉJÀ FAIT)
```
Pour chaque env (CartPole, CliffWalking, ...):
  - Config 1: St → St+1
  - Config 2: St → St+5
  - Config 3: ΔS
  Sauvegarder checkpoints
```

### **Phase 2: Entraîner les Precedence Networks** 🔨 (À FAIRE)
```
Pour chaque (env, wm_config):
  - Créer dataset (positif: vraies transitions, négatif: aléatoires)
  - Entraîner PN avec BCE loss
  - Early stopping
  - Sauvegarder checkpoints
```

### **Phase 3: Entraîner les RL Agents** 🔨 (À FAIRE)
```
Pour chaque (env, agent_type, with_precedence):
  - agent_type ∈ {DQN, PPO}
  - with_precedence ∈ {True, False}
  - Entraîner N épisodes
  - Enregistrer: reward, success_rate, exploration_efficiency
  - Sauvegarder checkpoints
```

### **Phase 4: Évaluation & Comparaison** 🔨 (À FAIRE)
```
Métriques:
  - Cumulative reward (moyenne + std)
  - Success rate
  - Learning speed (epochs to convergence)
  - Action validity rate (% actions validées par PN)
  - Sample efficiency (reward per environment step)
  
Plots:
  - Learning curves (WITH vs WITHOUT PN)
  - Comparaison DQN vs PPO
  - Effet de la PN config (1, 2, 3)
  - Comparaison multi-env
```

---

## 7️⃣ MÉTHODOLOGIE DE COMPARAISON

### **Baseline (SANS Precedence Network)**
```python
for episode in range(N_EPISODES):
    state = env.reset()
    for step in range(MAX_STEPS):
        # Agent décide action
        action = agent.select_action(state)
        
        # Exécuter directement dans l'env
        next_state, reward, done, _ = env.step(action)
        
        # Stocker et apprendre
        agent.remember(state, action, reward, next_state, done)
        agent.learn()
```

### **AVEC Precedence Network (Config 1: St → St+1)**
```python
for episode in range(N_EPISODES):
    state = env.reset()
    for step in range(MAX_STEPS):
        # Agent décide action
        action = agent.select_action(state)
        
        # Prédire SANS exécuter dans env réel
        predicted_next_state = world_model.predict(state, action)
        
        # Valider avec PN
        score = precedence_net(state, predicted_next_state)
        
        if score > THRESHOLD:
            # ✅ Action validée → Exécuter
            next_state, reward, done, _ = env.step(action)
            validation_success = True
        else:
            # ❌ Action rejetée → Chercher alternative
            alternative_action = agent.get_alternative()
            predicted_next_state = world_model.predict(state, alternative_action)
            next_state, reward, done, _ = env.step(alternative_action)
            validation_success = False
        
        # Stocker et apprendre
        agent.remember(state, action, reward, next_state, done)
        agent.learn()
        
        # Enregistrer validation rate
        metrics['validation_rate'].append(validation_success)
```

---

## 8️⃣ RÉSULTATS ATTENDUS

### **Si Precedence Network est utile:**
- ✅ Meilleure convergence (learning curves plus lisses)
- ✅ Higher final reward
- ✅ Better exploration efficiency
- ✅ Fewer invalid actions

### **Si Precedence Network n'aide pas:**
- ❌ Même performance (overhead computational)
- ❌ Latency due to PN inference

---

## 9️⃣ LIVRABLES FINAUX

```
thesis/
├── precedence_network_comparison.pdf
├── plots/
│   ├── learning_curves_dqn_cartpole.png
│   ├── learning_curves_ppo_cartpole.png
│   ├── comparison_with_vs_without_PN.png
│   ├── multi_env_comparison.png
│   └── ...
├── tables/
│   ├── results_summary.csv
│   ├── metrics_comparison.csv
│   └── ...
└── code/
    ├── trained_models/
    └── experiment_logs/
```

---

## 🚀 Ordre d'implémentation

1. **Créer PrecedenceNetwork class** (3 configs)
2. **Créer dataset generator** (positif/négatif)
3. **Entraîner PN** (for each env)
4. **Créer DQN agent** (with/without PN option)
5. **Créer PPO agent** (with/without PN option)
6. **Experiment runner** (orchestrate all)
7. **Evaluation & plots** (comparaison)
8. **Write thesis** (résultats + discussion)

---

## 📌 Notes importantes

- **World Models**: Déjà entraînés ✅
- **Precedence Network**: Novel contribution 🔨
- **RL Agents**: Standard DQN/PPO + modification pour PN
- **Multi-env**: Crucial for generalization
- **Reproducibility**: Fixer les seeds, sauvegarder configs

Commençons? 🎯
