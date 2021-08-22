import os
from abc import *


class BaseESLoop(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def run(self):
        pass
