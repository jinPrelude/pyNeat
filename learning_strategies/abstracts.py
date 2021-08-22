from abc import *


class BaseOffspringStrategy(metaclass=ABCMeta):
    @abstractmethod
    def __init__(selfm):
        pass

    @abstractmethod
    def get_elite_model(self):
        pass

    @abstractmethod
    def init_offspring(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass
