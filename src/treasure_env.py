import numpy as np
import random
from random import randint

class TreasureEnv():
    def __init__(self, size, seed):
        if seed is not None:
            random.seed(seed)

        self.size = size
        self.remain_step = -1
        self.num_traps = randint(0, (size*size)//2)

        self.__target_location = np.array([randint(0, self.size-1), randint(0, self.size-1)])
        self.agent_location  = np.array([-1, -1])

        self.traps = [[0 for _ in range(self.size)] for _ in range(self.size)]
        all_trap_positions = [(i, j) for i in range(self.size) for j in range(self.size) if not (i == self.__target_location[0] and j == self.__target_location[1])]
        random.shuffle(all_trap_positions)
        for trap_positions in all_trap_positions[:self.num_traps]:
            self.traps[trap_positions[0]][trap_positions[1]] = 1

        self.action_space = 4
        self.action_to_direction = {
            0: np.array([-1, 0]), # Up
            1: np.array([1, 0]),  # Down
            2: np.array([0, -1]), # Left
            3: np.array([0, 1]),  # Right
        }

    def reset(self):
        """
        Returns
            observation (agent and target location) and distance info
        """

        self.remain_step = self.size*self.size

        self.agent_location = np.array([randint(0, self.size-1), randint(0, self.size-1)])
        while self.traps[self.agent_location[0]][self.agent_location[1]] == 1 or np.array_equal(self.agent_location, self.__target_location):
            self.agent_location = np.array([randint(0, self.size-1), randint(0, self.size-1)])

        obs  = self.__get_obs()
        info = self.__get_info()

        return obs, info
    
    def step(self, action):
        """
        Returns
            obs, reward, terminated, info
        """
        self.remain_step -= 1

        self.agent_location = np.clip(self.agent_location + self.action_to_direction[action], 0, self.size-1)

        obs = self.__get_obs()
        info = self.__get_info()

        terminated = np.array_equal(self.agent_location, self.__target_location) or (self.traps[self.agent_location[0]][self.agent_location[1]] == 1)
        reward = 100 if (terminated and self.traps[self.agent_location[0]][self.agent_location[1]] != 1) else -1
        
        truncated = (self.remain_step == 0)

        return obs, reward, terminated, truncated, info
    
    def render(self):
        for i in range(self.size):
            for j in range(self.size):
                if np.array_equal([i, j], self.agent_location):
                    print('[O]', end='')
                elif np.array_equal([i, j], self.__target_location):
                    print('[X]', end='')
                elif self.traps[i][j] == 1:
                    print('[!]', end='')
                else:
                    print('[ ]', end='')
            print()

    def __get_obs(self):
        return {"agent": self.agent_location.copy(), "target": self.__target_location.copy()}
    
    def __get_info(self):
        return {"distance": self.__calc_distance()}

    def __calc_distance(self):
        """
        Returns
            Manhattan distance
        """
        return abs(self.agent_location[0] - self.__target_location[0]) + abs(self.agent_location[1] - self.__target_location[1])