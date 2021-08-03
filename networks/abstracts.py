from abc import *

from torch import nn


class BaseNetwork(metaclass=ABCMeta):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def save_model(self, path, model_name):
        pass

    @abstractmethod
    def load_model(self, path):
        pass


class EvolutionNetwork(BaseNetwork, nn.Module):
    @abstractmethod
    def zero_init(self):
        pass

    @abstractmethod
    def get_param_list(self):
        pass

    @abstractmethod
    def apply_param(self, param_lst: list):
        pass
