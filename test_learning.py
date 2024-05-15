import numpy as np
import random
from pettingzoo.utils.env import AECEnv
from custom_environment import env

class QLearningAgent:
    def __init__(self, env: AECEnv, agent, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.env = env
        n_observation = int(np.prod(env.observation_space(agent).shape))
        n_actions = np.prod(env.action_space(agent).n)
        self.q_table = np.zeros(n_observation, n_actions)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space(agent).sample()
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        print(state)
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, num_episodes):
        total_reward = 0
        completed_episodes = 0

        state, reward, termination, truncation, _ = self.env.last()

        while completed_episodes < num_episodes:

            total_reward += reward

            if termination or truncation:
                action = None
            elif isinstance(state, dict) and "action_mask" in state:
                action = random.choice(np.flatnonzero(state["action_mask"]).tolist())
            else:
                action = self.choose_action(state)
                self.env.step(action)
                next_state, reward, termination, truncation, _ = self.env.last()
                self.learn(state, action, reward, next_state)
                state = next_state

            completed_episodes += 1

# Usare l'ambiente Magent2
custom_env = env()
custom_env.reset()
for agent in custom_env.agent_iter():
    QAgent = QLearningAgent(custom_env, agent)
    QAgent.train(1000)
