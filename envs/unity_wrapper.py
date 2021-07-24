import gym
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)

gym.logger.set_level(40)


class UnityCollectAppleWrapper:
    def __init__(self, name, worker_id, max_step=None):
        self.name = name
        self.max_step = max_step

        channel = EngineConfigurationChannel()
        unityenv = UnityEnvironment("unityenvs/collectapple/game.x86_64", worker_id=worker_id, side_channels=[channel])
        channel.set_configuration_parameters(time_scale=20.0)
        self.env = UnityToGymWrapper(unityenv)
        self.curr_step = 0

    def reset(self):
        self.curr_step = 0
        return_list = {}
        transition = {}
        s = self.env.reset()
        transition["state"] = s
        return_list["0"] = transition
        return return_list

    def step(self, action):
        self.curr_step += 1
        return_list = {}
        transition = {}
        s, r, d, info = self.env.step(action["0"])
        if self.max_step != "None" and self.max_step is not None:
            if self.curr_step >= self.max_step or d:
                d = True
        transition["state"] = s
        transition["reward"] = r
        transition["done"] = d
        transition["info"] = info
        return_list["0"] = transition
        return return_list, r, d, info

    def get_agent_ids(self):
        return ["0"]

    def render(self):
        pass

    def close(self):
        self.env.close()
