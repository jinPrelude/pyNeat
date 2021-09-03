from typing import Tuple

import gym
import numpy as np
from .abstracts import BaseEnvWrapper

gym.logger.set_level(40)


class GymWrapper(BaseEnvWrapper):
    """Wrapper for OpenAI Gym.

    Attributes
    ----------
    env : gym.Env
        OpenAI Gym environment.
    max_step : int
        Max step of the episode.
    curr_step : int
        Current step of the episode.
    name : str
        Name of the environment.
    """

    def __init__(self, name: str, max_step: int = None, pomdp: bool = False):
        """GymWrapper init method.

        Parameters
        ----------
        name : str
            Gym environment name to build.
        max_step : int, optional
            Max step of the episode, by default None
        pomdp : bool, optional
            Set environment to pomdp mode, by default False

        Raises
        ------
        AssertionError
            If pomdp is requested in an environment that does not support.
        """
        self.env = gym.make(name)
        if pomdp:
            if "LunarLander" in name:
                print("POMDP LunarLander")
                self.env = LunarLanderPOMDP(self.env)
            elif "CartPole" in name:
                print("POMDP CartPole")
                self.env = CartPolePOMDP(self.env)
            else:
                raise AssertionError(f"{name} doesn't support POMDP.")
        self.max_step = max_step
        self.curr_step = 0
        self.name = name

    def reset(self) -> dict:
        """Reset environment and return initial states.

        Returns
        -------
        dict
            Dictionary which the key is agent's id and the value is state.
        """
        self.curr_step = 0
        return_list = {}
        transition = {}
        s = self.env.reset()
        transition["state"] = s
        return_list["0"] = transition
        return return_list

    def seed(self, seed: int):
        """Set random seed.

        Parameters
        ----------
        seed : int
            Random seed to initialize.
        """
        self.env.seed(seed)

    def step(self, action: dict) -> Tuple[dict, float, bool, dict]:
        """Get the action and return next MDP(states, reward, done, and other infos).


        Parameters
        ----------
        action : dict
            Dictionary which the key is agent's id and the value is action.

        Returns
        -------
        Tuple[dict, float, bool, dict]
            return_list : Dctionary containing information for each agent in this step.
            reward : Environmental Rewards in this step.
            d : If the episode is over.
            info : Additional informations in this step.
        """
        self.curr_step += 1
        return_list = {}
        transition = {}
        s, r, d, info = self.env.step(action["0"])
        if self.max_step != "None":
            if self.curr_step >= self.max_step or d:
                d = True
        transition["state"] = s
        transition["reward"] = r
        transition["done"] = d
        transition["info"] = info
        return_list["0"] = transition
        return return_list, r, d, info

    def get_agent_ids(self) -> list:
        """Return the agent ids requires for the episode.
        Since openai gym is single agent environment simply returns 0.

        Returns
        -------
        list
            List of agent ids.
        """
        return ["0"]

    def render(self, mode: str = None) -> np.array:
        """Render the episode.

        Parameters
        ----------
        mode : str
            Set the environment render mode(ex. "rgb_array" mode returns the rendered rgb array).

        Returns
        -------
        np.array
            Rendered rgb array. None if "rgb_array" mode is off.
        """
        return self.env.render(mode=mode)

    def close(self):
        """Close the environment."""
        self.env.close()


class LunarLanderPOMDP(gym.ObservationWrapper):
    """ObservationWrapper for LunarLander pomdp mode.

    Attributes
    ----------
    env : gym.Env
        Continuous or Descrete LunarLander environment.
    """

    def __init__(self, env):
        """LunarLanderPOMDP Init method.

        Parameters
        ----------
        env : gym.Env
            Continuous or Descrete LunarLander environment.
        """
        super().__init__(env)

    def observation(self, obs: np.array) -> np.array:
        """Return the observation with padded velocity informations.

        Parameters
        ----------
        obs : np.array
            Original observation from lunarlander environment.

        Returns
        -------
        np.array
            Modified observation with padded velocity informations.
        """
        # modify obs
        obs[2] = 0
        obs[3] = 0
        obs[5] = 0
        return obs


class CartPolePOMDP(gym.ObservationWrapper):
    """ObservationWrapper for CartPole pomdp mode.

    Attributes
    ----------
    env : gym.Env
        CartPole environment.
    """

    def __init__(self, env):
        """CartPolePOMDP Init method.

        Parameters
        ----------
        env : gym.Env
            CartPole environment.
        """
        super().__init__(env)

    def observation(self, obs):
        """Return the observation with padded velocity informations.

        Parameters
        ----------
        obs : np.array
            Original observation from lunarlander environment.

        Returns
        -------
        np.array
            Modified observation with padded velocity informations.
        """
        # modify obs
        obs[1] = 0
        obs[3] = 0
        return obs
