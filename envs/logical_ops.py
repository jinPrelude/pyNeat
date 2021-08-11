import math
import gym
import numpy as np
from numpy.lib.stride_tricks import DummyArray


class AndOps(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        self.name = "AndOps"
        self.idx = 0
        self.obs_list = [
            np.array([0, 0], np.float32),
            np.array([1, 0], np.float32),
            np.array([0, 1], np.float32),
            np.array([1, 1], np.float32),
        ]
        self.answer_list = [0, 0, 0, 1]

    def step(self, action):
        r = -abs(action["0"] - self.answer_list[self.idx])
        d = self.idx == 3
        return_list = {}
        transition = {}
        transition["state"] = self.obs_list[self.idx]
        transition["reward"] = r
        transition["done"] = d
        transition["info"] = {}
        return_list["0"] = transition
        self.idx += 1
        return return_list, r, d, None

    def reset(self):
        self.idx = 0
        return_list = {}
        transition = {}
        s = self.obs_list[self.idx]
        transition["state"] = s
        return_list["0"] = transition
        return return_list

    def render(self, mode="human", close=False):
        pass

    def get_agent_ids(self):
        return ["0"]
