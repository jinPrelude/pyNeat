import numpy as np
from networks.neat.feedforward import NeatNetwork, RecurrentNetwork
from networks.neat.genes import Genome

model = NeatNetwork(2, 1, False, 1, -30, 30)
model.init_genes()
model.reset()
input = np.array([[0.3, 0.2]])
test = model.forward(input)


def test_mutate():
    model = NeatNetwork(2, 1, False, 1, -30, 30)
    model.init_genes()
    print(model.model.node_evals)
    model.genome.mutate_weight()
    print(model.model.node_evals)
    model.model = RecurrentNetwork.create(model.genome)
    print(model.model.node_evals)


if __name__ == "__main__":
    test_mutate()
