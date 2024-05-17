import numpy as np
import random
from pettingzoo.utils.env import AECEnv

class QLearningAgent:
    def __init__(self, env: AECEnv, agent, learning_rate=0.8, gamma=0.95, epsilon=0.2, epsilon_decay=0.99, epsilon_min=0.01):
        self.env = env
        self.agent = agent
        n_observation = np.prod(env.state().shape[:2])
        n_x_coordinates = env.state().shape[0]
        n_y_coordinates = env.state().shape[1]
        n_actions = np.prod(env.action_space(agent).n)
        self.q_table = np.zeros((n_x_coordinates, n_y_coordinates, n_actions))
        self.learning_rate = learning_rate
        self.discount_factor = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table[0][1]

    def choose_action(self, state):
        state_coordinates = self.get_table_coordinates(state)

        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space(self.agent).sample()
        else:
            max_value = np.max(self.q_table[state_coordinates[0]][state_coordinates[1]])

            max_indices = np.where(self.q_table[state_coordinates[0]][state_coordinates[1]] == max_value)[0]
            return random.choice(max_indices)
        
    def get_table_coordinates(self, state):
        return np.round(state * (self.env.state().shape[0] - 1)).astype(int)

    def learn(self, state, action, reward, next_state):
        state_coordinates = self.get_table_coordinates(state)
        next_state_coordinates = self.get_table_coordinates(next_state)
        best_next_action = np.argmax(self.q_table[next_state_coordinates[0]][next_state_coordinates[1]])
        
        td_target = reward + self.discount_factor * self.q_table[next_state_coordinates[0]][next_state_coordinates[1]][best_next_action]

        td_error = td_target - self.q_table[state_coordinates[0]][state_coordinates[1]][action]
        
        self.q_table[state_coordinates[0]][state_coordinates[1]][action] += self.learning_rate * td_error

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_state(self, observation):
        return observation[0][0][-2:]

    def train(self, num_episodes):
        total_reward = 0
        completed_episodes = 0

        obs, reward, termination, truncation, _ = self.env.last()
        state = self.get_state(obs)

        while completed_episodes < num_episodes:

            total_reward += reward

            if termination or truncation:
                action = None
            elif isinstance(state, dict) and "action_mask" in state:
                action = random.choice(np.flatnonzero(state["action_mask"]).tolist())
            else:
                action = self.choose_action(state)
                self.env.step(action)
                next_obs, reward, termination, truncation, _ = self.env.last()
                next_state = self.get_state(next_obs)
                self.learn(state, action, reward, next_state)
                state = next_state
            print("Completed episode: ", completed_episodes, " with reward: ", reward)
            completed_episodes += 1

        print("Total reward after training: ", total_reward)