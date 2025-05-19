import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Definindo a rede neural simples
class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, n_items, input_dim=1, gamma=0.99, epsilon=0.1, lr=1e-3, batch_size=32, memory_size=1000):
        self.n_items = n_items
        self.input_dim = input_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size

        self.policy_net = QNetwork(input_dim, n_items)
        self.target_net = QNetwork(input_dim, n_items)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        self.memory = deque(maxlen=memory_size)
        self.update_target()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def encode_state(self, state: int):
        return torch.tensor([[state]], dtype=torch.float32)

    def choose_action(self, state: int):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_items - 1)
        with torch.no_grad():
            state_tensor = self.encode_state(state)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        state_batch = torch.tensor([[s] for s, _, _, _, _ in batch], dtype=torch.float32)
        action_batch = torch.tensor([a for _, a, _, _, _ in batch], dtype=torch.long)
        reward_batch = torch.tensor([r for _, _, r, _, _ in batch], dtype=torch.float32)
        next_state_batch = torch.tensor([[ns] for _, _, _, ns, _ in batch], dtype=torch.float32)
        done_batch = torch.tensor([d for _, _, _, _, d in batch], dtype=torch.float32)

        q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()
        next_q_values = self.target_net(next_state_batch).max(1)[0]
        target_q = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        loss = self.loss_fn(q_values, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, env, episodes=1000, target_update_freq=100):
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, r1, done, info = env.step(action)

                # Combinação ponderada das recompensas
                w1, w2 = 1.0, 0.5
                reward = w1 * r1 + w2 * info['r2']

                self.store_transition(state, action, reward, next_state, done)
                self.train_step()
                state = next_state
                total_reward += reward

            if episode % target_update_freq == 0:
                self.update_target()

            if episode % 100 == 0:
                print(f"[DQN] Episode {episode}, Total Reward: {total_reward:.2f}")
