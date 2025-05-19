# env/movielens_env.py
import gym
import numpy as np
from gym import spaces
from typing import Tuple, Dict

class MovieLensMultiObjectiveEnv(gym.Env):
    def __init__(self, user_data, item_data, rating_matrix):
        super().__init__()
        self.users = user_data
        self.items = item_data
        self.ratings = rating_matrix  # shape: (num_users, num_items)
        self.num_users, self.num_items = self.ratings.shape

        self.state_dim = 10  # podemos usar o embedding simples
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_items)

        self.current_user_id = None
        self.history = []

    def reset(self):
        self.current_user_id = np.random.randint(0, self.num_users)
        self.history = []

        user_vector = self._get_user_vector(self.current_user_id)
        return user_vector

    def step(self, action: int) -> Tuple[np.ndarray, Tuple[float, float], bool, Dict]:
        user_id = self.current_user_id
        rating = self.ratings[user_id, action]
        self.history.append((user_id, action, rating))

        # Objetivo 1: rating dado (engajamento)
        reward_1 = rating / 5.0  # normalizado

        # Objetivo 2: fairness (diferença entre gêneros até agora)
        male_ratings = [r for uid, _, r in self.history if self.users[uid]['gender'] == 'M']
        female_ratings = [r for uid, _, r in self.history if self.users[uid]['gender'] == 'F']

        mean_male = np.mean(male_ratings) if male_ratings else 0
        mean_female = np.mean(female_ratings) if female_ratings else 0
        reward_2 = -abs(mean_male - mean_female) / 5.0  # penaliza diferença (normalizado)

        # Próximo estado: (re)gera vetor do usuário
        state = self._get_user_vector(user_id)
        done = len(self.history) >= 10

        return state, (reward_1, reward_2), done, {}

    def _get_user_vector(self, user_id):
        # Vetor binário simples: [genero_M, genero_F, idade_norm, ...]
        user = self.users[user_id]
        gender = [1.0, 0.0] if user['gender'] == 'M' else [0.0, 1.0]
        age = [user['age'] / 100.0]
        return np.array(gender + age + [0.0] * (self.state_dim - len(gender) - 1), dtype=np.float32)

# utils/preprocessing.py
def generate_synthetic_movielens_data(num_users=100, num_items=50):
    users = [
        {"gender": np.random.choice(["M", "F"]), "age": np.random.randint(18, 60)}
        for _ in range(num_users)
    ]
    items = [{"genre": np.random.choice(["Action", "Drama", "Comedy"])} for _ in range(num_items)]
    ratings = np.random.randint(1, 6, size=(num_users, num_items))  # rating de 1 a 5
    return users, items, ratings
