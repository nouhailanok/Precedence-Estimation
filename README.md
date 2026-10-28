# Precedence Estimation

Projet d'expérimentation en **apprentissage par renforcement** et en **modélisation de dynamique**. L'objectif est d'étudier l'utilisation d'un *world model* — un modèle neuronal capable de prédire la transition d'état — afin d'améliorer ou d'analyser les politiques DQN et PPO dans des environnements Gymnasium.

Le dépôt contient :

- des agents de référence **DQN** et **PPO** sans world model ;
- la collecte de transitions dans **CartPole-v1** et **CliffWalking-v1** ;
- l'entraînement d'un réseau de neurones prédisant `Δs = s' - s` ;
- la comparaison de plusieurs architectures DNN ;
- des expériences DQN/PPO avec estimation de précédence (*precedence estimation*) ;
- des notebooks, checkpoints, métriques et graphiques associés aux expériences.

## Environnements étudiés

| Environnement | Type d'observation | Algorithmes |
|---|---|---|
| `CartPole-v1` | Vecteur continu de dimension 4 | DQN, PPO, world model |
| `CliffWalking-v1` | État discret encodé en one-hot sur 48 états | DQN, PPO, world model |

## Organisation du dépôt

```text
.
├── ABLATION STUDY/              # Études d'ablation
├── DQN_plots/                   # Résultats et graphiques DQN
├── PPO_plots/                   # Résultats et graphiques PPO
├── Evaluation/                  # Scripts et notebooks d'évaluation
├── checkpoints_compare/         # Checkpoints des architectures comparées
├── compare_plots/               # Graphiques de comparaison des DNN
├── collect_data/                # Scripts/notebooks de collecte de données
├── data/                        # Données collectées et notebooks associés
├── dnn predictor/               # Expériences de prédiction par DNN
├── winner_model_plots/          # Résultats du meilleur modèle DNN
├── world_model/                 # World models et politiques avec précédence
├── dqn_agent.py                 # Baseline DQN
├── ppo_agent.py                 # Baseline PPO
├── train_dnn_mixed_policy.py    # Entraînement du modèle de transition
├── dnn_compare_mixed_policy.py  # Comparaison de plusieurs architectures DNN
└── README.md
```

## Installation

Python 3.9 ou une version plus récente est recommandé.

```bash
git clone https://github.com/nouhailanok/Precedence-Estimation.git
cd Precedence-Estimation

python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate         # Windows

pip install numpy scipy matplotlib torch gymnasium
```

Pour exécuter les notebooks :

```bash
pip install jupyter
jupyter notebook
```

> Si une erreur apparaît lors de l'installation de PyTorch, installez la version adaptée à votre système depuis le site officiel de PyTorch. Le code sélectionne automatiquement le GPU CUDA lorsqu'il est disponible, sinon il utilise le CPU.

## Utilisation

### 1. Baseline DQN

Le fichier `dqn_agent.py` entraîne un agent DQN avec :

- réseau Q multilayer perceptron ;
- replay buffer ;
- réseau cible ;
- exploration epsilon-greedy ;
- évaluation gloutonne sur 30 épisodes ;
- sauvegarde des métriques, graphiques et poids du modèle.

Entraîner sur CartPole :

```bash
python dqn_agent.py
```

Entraîner sur CliffWalking :

```bash
python dqn_agent.py --env CliffWalking-v1
```

Les résultats sont écrits dans :

```text
DQN_plots/<nom_de_l_environnement>/
```

On y trouve notamment la courbe d'apprentissage, les métriques JSON et le checkpoint `dqn_agent.pth`.

### 2. Baseline PPO

Le fichier `ppo_agent.py` implémente une baseline PPO avec :

- actor et critic séparés ;
- estimation généralisée de l'avantage — GAE ;
- objectif PPO avec clipping ;
- arrêt anticipé basé sur la divergence KL ;
- évaluation gloutonne et sauvegarde des résultats.

Entraîner sur CartPole :

```bash
python ppo_agent.py
```

Entraîner sur CliffWalking :

```bash
python ppo_agent.py --env CliffWalking-v1
```

Les résultats sont écrits dans :

```text
PPO_plots/<nom_de_l_environnement>/
```

### 3. Collecte des données

Les notebooks présents dans `data/` permettent de collecter les transitions de CartPole et CliffWalking :

- `data/data_collector_CartPole.ipynb`
- `data/data_collector_CliffWalking.ipynb`

Une transition contient généralement un état `s`, une action `a` et l'état suivant `s'`. Pour le modèle de transition CartPole, les états sont normalisés et les actions sont représentées en one-hot.

### 4. Entraînement du modèle de transition DNN

Le script `train_dnn_mixed_policy.py` entraîne un réseau qui prédit le déplacement d'état plutôt que l'état suivant directement :

```text
Δs = s' - s
s' prédit = s + Δs prédit
```

#### 4.1 Procédures de sélection selon l'environnement

Les expériences de prédiction de transition comparent plusieurs formulations de cible :

| Approche | Entrée | Cible | Perte |
|---|---|---|---|
| A — État complet à 1 pas | `s_t + a_t` | `s_{t+1}` | MSE |
| B — Delta à 1 pas | `s_t + a_t` | `s_{t+1} - s_t` | MSE |
| C — Delta à 1 pas robuste | `s_t + a_t` | `s_{t+1} - s_t` | Huber |
| D — État complet à n pas | `s_t + [a_t, ..., a_{t+n-1}]` | `s_{t+n}` | MSE |
| E — Delta à n pas | `s_t + [a_t, ..., a_{t+n-1}]` | `s_{t+n} - s_t` | Huber |

En formulation delta, on prédit d'abord le déplacement puis on reconstruit l'état futur :

```text
Δs prédit = modèle(s_t, a_t, ...)
s' prédit = s_t + Δs prédit
```

##### CartPole-v1

Pour `CartPole-v1`, les approches A à E sont comparées. Les configurations multi-pas testent `n ∈ {3, 5, 10}` et le meilleur horizon est sélectionné via l'erreur de validation. Les observations sont des vecteurs continus normalisés de dimension 4, et les actions sont encodées en one-hot avec 2 actions.

##### CliffWalking-v1

Pour `CliffWalking-v1`, l'entraînement se concentre directement sur la prédiction delta pour `n ∈ {1, 2, 3, 5, 8}`. Les états sont encodés en one-hot sur 48 cellules et l'environnement comporte 4 actions. Le meilleur horizon est choisi selon la MSE de validation puis évalué sur l'ensemble de test.

| Élément | CartPole-v1 | CliffWalking-v1 |
|---|---|---|
| Type d'état | Continu (4D) | Discret (48 cellules) |
| Encodage des états | Vecteur normalisé | One-hot (48) |
| Nombre d'actions | 2 | 4 |
| Stratégies | Comparaison A–E (état complet et delta) | Delta uniquement |
| Horizons testés | `n ∈ {3, 5, 10}` (pour les variantes multi-pas) | `n ∈ {1, 2, 3, 5, 8}` |
| Critère de sélection | Erreur de validation | MSE de validation |

Ces différences d'horizons et de sélection sont intentionnelles : CartPole compare d'abord les formulations de cibles (A–E), tandis que CliffWalking se concentre sur la prédiction delta à travers plusieurs fenêtres temporelles.

Le réseau reçoit un état de dimension 4 et une action one-hot de dimension 2, soit une entrée de dimension 6. Il prédit les quatre composantes de `Δs` correspondant à :

- position du chariot `x` ;
- vitesse `ẋ` ;
- angle du poteau `θ` ;
- vitesse angulaire `θ̇`.

Avant de lancer le script, vérifiez que le fichier de données attendu existe à la racine du dépôt :

```text
cartpole_data_mixed_policy.npz
```

Puis exécutez :

```bash
python train_dnn_mixed_policy.py
```

Le script :

1. charge les ensembles train, validation et test ;
2. entraîne le modèle avec MSE ou Huber loss ;
3. applique un scheduler de learning rate ;
4. utilise l'early stopping ;
5. évalue les prédictions dans l'espace normalisé et physique ;
6. réalise des évaluations en rollout ;
7. génère des graphiques dans `winner_model_plots/` ;
8. sauvegarde le modèle dans `transition_dnn_final.pth`.

### 5. Comparaison des architectures DNN

Le script `dnn_compare_mixed_policy.py` compare plusieurs architectures et plusieurs seeds :

- réseaux petits, moyens et larges ;
- réseaux plus profonds ;
- activations ELU ;
- BatchNorm ;
- Dropout.

Lancer la comparaison :

```bash
python dnn_compare_mixed_policy.py
```

Les checkpoints individuels sont enregistrés dans `checkpoints_compare/` et les figures dans `compare_plots/`. Les critères de comparaison incluent :

- MSE sur `Δs` ;
- RMSE dans les unités physiques ;
- MAE ;
- coefficient de détermination `R²` ;
- accuracy avec une tolérance `ε` ;
- nombre d'époques et nombre de paramètres.

## World model et précédence

Le dossier `world_model/` contient les expériences qui combinent les politiques de renforcement avec une estimation de transition. On y trouve notamment des notebooks pour :

- DQN avec précédence sur CartPole ;
- DQN avec précédence sur CliffWalking ;
- PPO avec précédence ;
- entraînement et utilisation de world models ;
- comparaison des résultats et visualisation des trajectoires.

Les notebooks principaux incluent :

- `world_model/world_model_CartPole.ipynb`
- `world_model/world_model_CliffWalking.ipynb`
- `world_model/dqn_with_precedence_CartPole_full.ipynb`
- `world_model/dqn_precedence_CliffWalking.ipynb`
- `world_model/ppo_with_precedence_CartPole.ipynb`
- `world_model/ppo_precedence_CliffWalking.ipynb`

Les checkpoints et les graphiques de ces expériences sont organisés dans des sous-dossiers dédiés du répertoire `world_model/`.

## Reproductibilité

Les expériences utilisent principalement la seed `42`. Pour reproduire les résultats au plus près :

- utilisez les mêmes versions de Python et des dépendances ;
- conservez les seeds configurées dans les scripts ;
- vérifiez que les données d'entraînement sont identiques ;
- utilisez le même périphérique de calcul lorsque cela est possible ;
- exécutez les notebooks dans l'ordre lorsque les résultats d'une étape sont utilisés par une autre.

Les scripts choisissent automatiquement entre CPU et GPU :

```python
torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Sorties générées

Selon l'expérience exécutée, le projet peut produire :

- des courbes de récompense et de loss ;
- des métriques au format JSON ;
- des checkpoints PyTorch `.pth` ;
- des prédictions réel/prédit ;
- des distributions d'erreur ;
- des graphiques de dérive (*rollout drift*) ;
- des heatmaps et radars de comparaison d'architectures.

## Dépendances principales

- [Python](https://www.python.org/)
- [PyTorch](https://pytorch.org/)
- [Gymnasium](https://gymnasium.farama.org/)
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Jupyter](https://jupyter.org/)

## Remarques

- Les scripts DQN et PPO constituent des baselines sans world model.
- Les chemins de données sont définis directement dans les scripts ; adaptez `DATA_PATH` si vos fichiers `.npz` se trouvent dans un autre dossier.
- Les expériences peuvent être coûteuses en temps de calcul, notamment la comparaison de plusieurs architectures avec plusieurs seeds.
- Les notebooks peuvent générer de nombreux fichiers de sortie ; vérifiez les dossiers de sortie avant de relancer une expérience.

## Licence

Aucune licence open source n'est actuellement spécifiée dans le dépôt. Ajoutez un fichier `LICENSE` si vous souhaitez définir les conditions de réutilisation du projet.

## Auteur

Projet développé par [nouhailanok](https://github.com/nouhailanok).
