import numpy as np

from networks.neat.feedforward import NeatNetwork

model = NeatNetwork(2, 1, True)
input = np.array([0.3, 0.2])
test = model.forward(input)