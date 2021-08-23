from abc import *


class BaseNeat(metaclass=ABCMeta):
    def __init__(self):
        super(BaseNeat, self).__init__()

    @abstractmethod
    def init_genome(self):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def replace_genome(self):
        pass

    @abstractmethod
    def save_model(save_path, model_name):
        pass

    @abstractmethod
    def load_model(path):
        pass

    @abstractmethod
    def mutate(path):
        pass

    @abstractmethod
    def crossover(spouse, draw):
        pass
