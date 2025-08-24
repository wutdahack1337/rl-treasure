import random
import numpy as np
from random import randint
from collections import defaultdict

class TreasureAgent:
    def __init__(self, env, learning_rate, epsilon, discount_factor, seed):
        if seed is not None:
            random.seed(seed)

        self.env = env

        self.epsilon = epsilon

        self.q_values = defaultdict(lambda: np.zeros(self.env.action_space))
        self.learning_rate   = learning_rate
        self.discount_factor = discount_factor

    def get_action(self, obs, verbose = 1):
        obs = tuple(np.concatenate(list(obs.values())))

        if random.random() < self.epsilon:
            if verbose == 1:
                print("random action")

            return randint(0, self.env.action_space-1)
        else:
            if verbose == 1:
                print("greedy action")

            return int(np.argmax(self.q_values[obs]))
    
    def learn(self, obs, action, reward, terminated, next_obs):
        obs      = tuple(np.concatenate(list(obs.values())))
        next_obs = tuple(np.concatenate(list(next_obs.values())))
    
        future_q_value = (not terminated)*np.max(self.q_values[next_obs])
        target = reward + self.discount_factor*future_q_value
        temporal_difference = target - self.q_values[obs][action]
        self.q_values[obs][action] = self.q_values[obs][action] + self.learning_rate*temporal_difference