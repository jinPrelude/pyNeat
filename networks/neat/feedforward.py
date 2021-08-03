# Modified https://github.com/CodeReclaimers/neat-python/blob/master/neat/nn/feed_forward.py.

from abc import *
import os
import pickle
import math

import numpy as np

from networks.abstracts import BaseNetwork
from networks.neat.genes import Genome


class NeatNetwork(BaseNetwork):
    def __init__(self, num_state, num_action, discrete_action, mutate_sigma, max_weight, min_weight):
        self.num_state = num_state
        self.num_action = num_action
        self.discrete_action = discrete_action

        # for genome
        self.mutate_sigma = mutate_sigma
        self.max_weight = max_weight
        self.min_weight = min_weight

    def init_genes(self):
        self.genome = Genome(self.num_state, self.num_action, self.mutate_sigma, self.max_weight, self.min_weight)
        self.model = RecurrentNetwork.create(self.genome)

    def forward(self, x):
        output = self.model.activate(x[0])
        if self.discrete_action:
            output = np.argmax(output)
        return output

    def normal_init(self, mu, std):
        self.genome.normal_init(mu, std)
        self.model = RecurrentNetwork.create(self.genome)

    def reset(self):
        self.model.reset()

    def update_model(self, nodes, connections):
        self.genome.update_genome(nodes, connections)
        self.model = RecurrentNetwork.create(self.genome)

    def save_model(self, save_path, model_name):
        model_name += ".pkl"
        save_path = os.path.join(save_path, model_name)
        with open(save_path, "wb") as f:
            pickle.dump([self.genome, self.model.node_evals], f, pickle.HIGHEST_PROTOCOL)
        return save_path

    def load_model(self, path):
        with open(path, "rb") as f:
            test = pickle.load(f)
        self.genome = test[0]
        self.model = RecurrentNetwork.create(self.genome)

    def mutate(self):
        self.genome.mutate_weight()
        self.genome.mutate_add_node()
        self.model = RecurrentNetwork.create(self.genome)


class RecurrentNetwork(object):
    def __init__(self, inputs, outputs, node_evals):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals

        self.values = [{}, {}]
        for v in self.values:
            for k in list(inputs) + list(outputs):
                v[k] = 0.0

            for node, _, _, links in self.node_evals:
                v[node] = 0.0
                for i, w in links:
                    v[i] = 0.0
        self.active = 0

    def reset(self):
        self.values = [dict((k, 0.0) for k in v) for v in self.values]
        self.active = 0

    def activate(self, inputs):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        ivalues = self.values[self.active]
        ovalues = self.values[1 - self.active]
        self.active = 1 - self.active

        for i, v in zip(self.input_nodes, inputs):
            ivalues[i] = v
            ovalues[i] = v

        for node, activation, bias, links in self.node_evals:
            node_inputs = [ivalues[i] * w for i, w in links]
            ovalues[node] = activation(bias + sum(node_inputs))

        return np.array([ovalues[i] for i in self.output_nodes])

    @staticmethod
    def create(genome):
        """Receives a genome and returns its phenotype (a RecurrentNetwork)."""
        connect_genes = genome.get_connect_genes()
        connections = [cg.connection for cg in connect_genes.values() if cg.enabled]
        sensor_nodes = genome.get_sensor_nodes()
        output_nodes = genome.get_output_nodes()
        required = required_for_output(sensor_nodes, output_nodes, connections)

        # Gather inputs and expressed connections.
        node_inputs = {}
        for cg in connect_genes.values():
            if not cg.enabled:
                continue

            in_node, out_node = cg.connection
            if out_node not in required and in_node not in required:
                continue

            if out_node not in node_inputs:
                node_inputs[out_node] = [(in_node, cg.weight)]
            else:
                node_inputs[out_node].append((in_node, cg.weight))

        genome_nodes = genome.get_nodes()
        node_evals = []
        for node_key, inputs in node_inputs.items():
            node = genome_nodes[node_key]
            activation_function = math.tanh
            # aggregation_function = genome_config.aggregation_function_defs.get(node.aggregation) # summation
            node_evals.append((node_key, activation_function, node.bias, inputs))

        return RecurrentNetwork(sensor_nodes, output_nodes, node_evals)


def required_for_output(inputs, outputs, connections):
    """
    Collect the nodes whose state is required to compute the final network output(s).
    :param inputs: list of the input identifiers
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.
    NOTE: It is assumed that the input identifier set and the node identifier set are disjoint.
    By convention, the output node ids are always the same as the output index.

    Returns a set of identifiers of required nodes.
    """

    required = set(outputs)
    s = set(outputs)
    while 1:
        # Find nodes not in S whose output is consumed by a node in s.
        t = set(a for (a, b) in connections if b in s and a not in s)

        if not t:
            break

        layer_nodes = set(x for x in t if x not in inputs)
        if not layer_nodes:
            break

        required = required.union(layer_nodes)
        s = s.union(t)

    return required


def feed_forward_layers(inputs, outputs, connections):
    """
    Collect the layers whose members can be evaluated in parallel in a feed-forward network.
    :param inputs: list of the network input nodes
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.

    Returns a list of layers, with each layer consisting of a set of node identifiers.
    Note that the returned layers do not contain nodes whose output is ultimately
    never used to compute the final network output.
    """

    required = required_for_output(inputs, outputs, connections)

    layers = []
    s = set(inputs)
    while 1:
        # Find candidate nodes c for the next layer.  These nodes should connect
        # a node in s to a node not in s.
        c = set(b for (a, b) in connections if a in s and b not in s)
        # Keep only the used nodes whose entire input set is contained in s.
        t = set()
        for n in c:
            if n in required and all(a in s for (a, b) in connections if b == n):
                t.add(n)

        if not t:
            break

        layers.append(t)
        s = s.union(t)

    return layers


# class FeedForwardNetwork(object):
#     def __init__(self, inputs, outputs, node_evals):
#         self.input_nodes = inputs
#         self.output_nodes = outputs
#         self.node_evals = node_evals
#         self.values = dict((key, 0.0) for key in inputs + outputs)

#     def activate(self, inputs):
#         if len(self.input_nodes) != len(inputs):
#             raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

#         for k, v in zip(self.input_nodes, inputs):
#             self.values[k] = v

#         for node, act_func, bias, links in self.node_evals:
#             node_inputs = []
#             for i, w in links:
#                 node_inputs.append(self.values[i] * w)
#             self.values[node] = act_func(bias + sum(node_inputs))

#         return np.array([self.values[i] for i in self.output_nodes])

#     @staticmethod
#     def create(genome):
#         """Receives a genome and returns its phenotype (a FeedForwardNetwork)."""

#         # Gather expressed connections.
#         connect_genes = genome.get_connect_genes(key="connection")
#         connections = [cg.connection for cg in connect_genes.values() if cg.enabled]
#         sensor_nodes = genome.get_sensor_nodes()
#         output_nodes = genome.get_output_nodes()
#         layers = feed_forward_layers(sensor_nodes, output_nodes, connections)
#         genome_nodes = genome.get_nodes()
#         node_evals = []
#         for layer in layers:
#             for node in layer:
#                 inputs = []
#                 node_expr = []  # currently unused
#                 for conn_key in connections:
#                     inode, onode = conn_key
#                     if onode == node:
#                         cg = connect_genes[conn_key]
#                         inputs.append((inode, cg.weight))
#                         node_expr.append("v[{}] * {:.7e}".format(inode, cg.weight))

#                 ng = genome_nodes[node]
#                 activation_function = math.tanh
#                 node_evals.append((node, activation_function, ng.bias, inputs))

#         return FeedForwardNetwork(sensor_nodes, output_nodes, node_evals)
