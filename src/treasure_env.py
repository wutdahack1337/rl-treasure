import numpy as np
import random
from random import randint

class TreasureEnv():
    def __init__(self, size, seed):
        """
        Init (i, j) world
        """
        if seed is not None:
            random.seed(seed)

        self.size = size
        self.remain_step = -1
        self.num_traps = randint(1, size*size//2)

        self.__target_location = np.array([randint(0, self.size-1), randint(0, self.size-1)])
        self.agent_location  = np.array([-1, -1])
        self.traps = []
        for i in range(self.num_traps):
            trap_location = np.array([randint(0, self.size-1), randint(0, self.size-1)])
            while np.array_equal(trap_location, self.__target_location):
                trap_location = np.array([randint(0, self.size-1), randint(0, self.size-1)])
            self.traps.append(trap_location)

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
        while self._check_trap_location(self.agent_location):
            self.agent_location = np.array([randint(0, self.size-1), randint(0, self.size-1)])
        
        while np.array_equal(self.agent_location, self.__target_location):
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

        self.agent_location += self.action_to_direction[action]

        obs = self.__get_obs()
        info = self.__get_info()
        
        terminated = np.array_equal(self.agent_location, self.__target_location)
        reward = 100 if terminated else -1

        for i in range(self.num_traps):
            if np.array_equal(self.agent_location, self.traps[i]):
                terminated = True
                reward = -1000

        truncated = (self.remain_step == 0)

        return obs, reward, terminated, truncated, info
    
    def render(self):
        for i in range(self.size):
            for j in range(self.size):
                if np.array_equal([i, j], self.agent_location):
                    print('[O]', end='')
                elif np.array_equal([i, j], self.__target_location):
                    print('[X]', end='')
                elif self._check_trap_location([i, j]):
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

    def _check_trap_location(self, location):
        for i in range(self.num_traps):
            if np.array_equal(location, self.traps[i]):
                return True
        return False


