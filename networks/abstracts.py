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
