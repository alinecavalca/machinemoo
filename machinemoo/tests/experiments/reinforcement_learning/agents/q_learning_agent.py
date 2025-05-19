import numpy as np
import random
from collections import defaultdict
from machinemoo.tests.experiments.reinforcement_learning.envs.movielens_env import SyntheticFairRecommenderEnv

class QLearningAgent:
    def __init__(self, n_items: int, n_groups: int, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1):
        """
        Agente Q-learning para otimização de recomendação justa.
        
        Parameters:
        - n_items: número de itens recomendáveis.
        - n_groups: número de grupos (masculino, feminino).
        - alpha: taxa de aprendizado.
        - gamma: fator de desconto.
        - epsilon: taxa de exploração.
        """
        self.n_items = n_items
        self.n_groups = n_groups
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(self.n_items))  # Tabela Q

    def choose_action(self, state: int) -> int:
        """
        Escolhe uma ação (item) para recomendar usando uma política epsilon-greedy.
        
        Parameters:
        - state: estado atual (grupo do usuário).
        
        Returns:
        - Ação escolhida (índice do item recomendado).
        """
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.n_items - 1)  # Exploração: escolha aleatória
        else:
            return np.argmax(self.q_table[state])  # Exploração: escolha o melhor item

    def update_q_table(self, state: int, action: int, reward: float, next_state: int):
        """
        Atualiza a tabela Q com base no valor obtido após a ação.
        
        Parameters:
        - state: estado atual (grupo do usuário).
        - action: item recomendado.
        - reward: recompensa recebida.
        - next_state: próximo estado (grupo do próximo usuário).
        """
        best_next_action = np.argmax(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (reward + self.gamma * self.q_table[next_state][best_next_action] - self.q_table[state][action])

    def train(self, env, episodes: int = 1000):
        """
        Treina o agente Q-learning no ambiente fornecido.
        
        Parameters:
        - env: o ambiente de recomendação (instância de SyntheticFairRecommenderEnv).
        - episodes: número de episódios de treinamento.
        """
        for episode in range(episodes):
            state = env.reset()  # Resetar o ambiente e obter o estado inicial
            done = False
            total_reward = 0
            
            while not done:
                action = self.choose_action(state)  # Escolher uma ação (item recomendado)
                next_state, r1, done, info = env.step(action)  # Realizar a ação e obter o feedback
                
                # Combinação ponderada de recompensas: r = w1 * r1 + w2 * r2
                w1, w2 = 1.0, 0.5  # Pesos para satisfação e justiça (ajustáveis)
                reward = w1 * r1 + w2 * info['r2']  # Recompensa combinada

                self.update_q_table(state, action, reward, next_state)  # Atualizar Q-table

                state = next_state  # Mover para o próximo estado
                total_reward += reward

            if episode % 100 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward:.2f}")

# Exemplo de uso:

if __name__ == "__main__":
    env = SyntheticFairRecommenderEnv(group_prob=0.5)  # Inicializar o ambiente
    agent = QLearningAgent(n_items=5, n_groups=2)  # Inicializar o agente Q-learning
    agent.train(env, episodes=1000)  # Treinar o agente
