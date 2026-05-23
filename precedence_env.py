import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PrecedenceEnv(gym.Env):
    """
    Environnement d'ordonnancement de tâches avec contraintes de précédence.
    """
    def __init__(self, num_tasks=5):
        super().__init__()
        self.num_tasks = num_tasks
        
        # Action : choisir l'index d'une tâche à exécuter (de 0 à num_tasks - 1)
        self.action_space = spaces.Discrete(self.num_tasks)
        
        # Observation : Un dictionnaire contenant le graphe et le statut
        self.observation_space = spaces.Dict({
            # Matrice d'adjacence (NxN) : 1 si i doit précéder j
            "adj_matrix": spaces.Box(low=0, high=1, shape=(num_tasks, num_tasks), dtype=np.int8),
            # Tâches terminées (N,) : 1 si fini, 0 sinon
            "completed": spaces.Box(low=0, high=1, shape=(num_tasks,), dtype=np.int8),
            # Tâches prêtes (N,) : 1 si exécutable maintenant, 0 sinon
            "ready": spaces.Box(low=0, high=1, shape=(num_tasks,), dtype=np.int8)
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        
        # 1. Génération d'un DAG aléatoire (Directed Acyclic Graph)
        self.adj_matrix = np.zeros((self.num_tasks, self.num_tasks), dtype=np.int8)
        for i in range(self.num_tasks):
            for j in range(i + 1, self.num_tasks): # i+1 garantit l'absence de cycles
                if self.np_random.random() < 0.3:  # 30% de chance d'avoir une dépendance
                    self.adj_matrix[i, j] = 1
                    
        # 2. Initialisation des statuts
        self.completed = np.zeros(self.num_tasks, dtype=np.int8)
        self._update_ready_tasks()
        
        return self._get_obs(), {}

    def _update_ready_tasks(self):
        """Met à jour le vecteur 'ready' en lisant le graphe et les tâches terminées."""
        self.ready = np.zeros(self.num_tasks, dtype=np.int8)
        for j in range(self.num_tasks):
            if self.completed[j] == 0:
                # Chercher tous les parents de la tâche j
                parents = np.where(self.adj_matrix[:, j] == 1)[0]
                # Si TOUS les parents sont terminés, la tâche est prête
                if np.all(self.completed[parents] == 1):
                    self.ready[j] = 1

    def _get_obs(self):
        """Retourne l'état actuel sous forme de dictionnaire."""
        return {
            "adj_matrix": self.adj_matrix.copy(),
            "completed": self.completed.copy(),
            "ready": self.ready.copy()
        }

    def step(self, action):
        reward = 0
        
        # RÈGLE 1 : L'action est-elle valide ?
        if self.ready[action] == 1:
            # Succès : on marque comme terminée et on actualise
            self.completed[action] = 1
            self._update_ready_tasks()
            reward = -1.0  # Pénalité standard de temps
        else:
            # Échec : action illégale
            reward = -10.0 # Grosse pénalité, l'état ne change pas

        # RÈGLE 2 : Le jeu est-il fini ?
        terminated = bool(np.all(self.completed == 1))
        truncated = False # Pas de limite de temps artificielle ici

        return self._get_obs(), reward, terminated, truncated, {}