from abc import *

import numpy as np


class BaseEnvWrapper(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractclassmethod
    def reset():
        pass

    @abstractclassmethod
    def step(action: np.array) -> tuple:
        pass

    @abstractclassmethod
    def get_agent_ids() -> dict:
        pass

    @abstractclassmethod
    def render() -> np.array:
        pass

    @abstractclassmethod
    def close():
        pass
