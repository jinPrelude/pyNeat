import numpy as np

from learning_strategies.neat.neat_utils import count
from networks.neat.feedforward import NeatNetwork
from networks.neat.genes import Genome

model = NeatNetwork(2, 1, False, 1, -30, 30)
model.init_genes()
model.reset()
input = np.array([[0.3, 0.2]])
test = model.forward(input)
print(test)
