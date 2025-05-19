from machinemoo.tests.experiments.reinforcement_learning.envs.movielens_env import SyntheticFairRecommenderEnv
from machinemoo.tests.experiments.reinforcement_learning.agents.dqn_agent import DQNAgent

if __name__ == "__main__":
    env = SyntheticFairRecommenderEnv(group_prob=0.5)
    agent = DQNAgent(n_items=5)
    agent.train(env, episodes=1000)
