# This code is brought from neat-python and modified to fit our repo.
# https://github.com/CodeReclaimers/neat-python/blob/master/neat/nn/feed_forward.py.

import os
import pickle
import math
from copy import deepcopy
import random

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from networks.neat.genes import Genome
from .utils import find_required_nodes
from .abstracts import BaseNeat


class NeatNetwork(BaseNeat):
    def __init__(
        self,
        num_state,
        num_action,
        discrete_action,
        init_mu,
        init_std,
        mutate_std,
        max_weight,
        min_weight,
        probs,
    ):
        self.num_state = num_state
        self.num_action = num_action
        self.discrete_action = discrete_action
        self.init_mu = init_mu
        self.init_std = init_std
        self.mutate_std = mutate_std

        # for genome
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.probs = probs

    def init_genome(self):
        self.genome = Genome(
            self.num_state,
            self.num_action,
            self.init_mu,
            self.init_std,
            self.mutate_std,
            self.max_weight,
            self.min_weight,
        )
        self.genome.normal_init()
        self._update_model()  # model must be updated after genome modified.

    def forward(self, x):
        output = self.model.activate(x[0])
        if self.discrete_action:
            output = np.argmax(output)
        return output

    def reset(self):
        self.model.reset()

    def _update_model(self):
        self.model = RecurrentNetwork.create(self.genome)

    # TODO: it works dirty.
    def replace_genome(self, nodes, connections):
        self.genome.replace_genome(nodes, connections)
        self._update_model()  # model must be updated after genome modified.

    def save_model(self, save_path, model_name):
        save_path = os.path.join(save_path, model_name)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        save_model_path = os.path.join(save_path, model_name + ".pkl")
        with open(save_model_path, "wb") as f:
            pickle.dump(self.genome, f, pickle.HIGHEST_PROTOCOL)
        draw_path = self._draw_network(save_path, model_name)
        return [save_model_path, draw_path]

    def load_model(self, path):
        with open(path, "rb") as f:
            self.genome = pickle.load(f)
        self._update_model()  # model must be updated after genome modified.

    def _draw_network(self, save_path, model_name):
        color_map = []
        connections = self.genome.get_connect_genes()
        nodes = self.genome.get_nodes()

        g_pos = nx.complete_multipartite_graph()
        g = nx.DiGraph()

        enabled_nodes = set()
        for connection in connections.values():
            if connection.enabled:
                in_node = connection.in_node_num
                out_node = connection.out_node_num
                g.add_edge(in_node, out_node)
                g_pos.add_edge(in_node, out_node)
                enabled_nodes.add(in_node)
                enabled_nodes.add(out_node)
        for node_num in g_pos.nodes:
            if nodes[node_num].type == "sensor":
                color_map.append("#00d2d9")
                g_pos.nodes[node_num]["type"] = 0
            elif nodes[node_num].type == "output":
                color_map.append("#d96900")
                g_pos.nodes[node_num]["type"] = 2
            else:
                color_map.append("#7b00d9")
                g_pos.nodes[node_num]["type"] = 1

        pos = nx.multipartite_layout(g_pos, subset_key="type")
        nx.draw(g, with_labels=True, pos=pos, node_color=color_map)
        saved_path = os.path.join(save_path, model_name + f"_graph.png")
        plt.savefig(saved_path)
        plt.clf()
        return saved_path

    def _check_genome_model_synced(self):
        # check if genome.nodes == model nodes
        nodes = self.genome.node_genes.nodes
        all_node_keys = set(nodes.keys())
        connections = self.genome.connect_genes.connections
        connections_nodes = set()
        for input_node_num, output_node_num in connections.keys():
            connections_nodes.add(input_node_num)
            connections_nodes.add(output_node_num)
        node_synced = connections_nodes == all_node_keys

        # check if node_evals <= genome nodes
        node_evals = self.model.node_evals
        node_eval_node_nums = set()
        for node in node_evals:
            curr_node, act_func, bias, input_list = node
            node_eval_node_nums.add(curr_node)
            assert nodes[curr_node].bias == bias
            for input_node_num, weight in input_list:
                assert connections[(input_node_num, curr_node)].weight == weight
                node_eval_node_nums.add(input_node_num)
        eval_node_synced = node_eval_node_nums <= all_node_keys

        return node_synced and eval_node_synced

    def mutate(self):
        self.genome.mutate_weight(self.probs["mutate_weight"])
        self.genome.mutate_bias(self.probs["mutate_weight"])
        self.genome.mutate_add_node(self.probs["mutate_add_node"])
        self.genome.mutate_add_connection(self.probs["mutate_add_connection"])
        self._update_model()  # model must be updated after genome modified.
        assert self._check_genome_model_synced()

    def crossover(self, spouse, draw=False):
        p1_connect_genes = self.genome.get_connect_genes()
        p2_connect_genes = spouse.genome.get_connect_genes()
        p1_connections = set(self.genome.get_connect_genes().keys())
        p2_connections = set(spouse.genome.get_connect_genes().keys())

        child = deepcopy(self)
        child_nodes = {}
        child_connections = {}

        # matching genes crossover
        matching_connections = p1_connections & p2_connections
        child_nodes.update(find_required_nodes(matching_connections, self.genome))
        for connection in matching_connections:
            rand_num = random.random()
            if rand_num > 0.5:
                child_connections[connection] = p1_connect_genes[connection]
            else:
                child_connections[connection] = p2_connect_genes[connection]
            if (
                p1_connect_genes[connection].enabled == False
                and p2_connect_genes[connection].enabled == False
            ):
                if random.random() < self.probs["re_enable"]:
                    child_connections[connection].enabled = True

        # disjoint & excess crossover(treat the two equally).
        def _add_node_connections(
            child_nodes, child_connections, connection_keys, parent_genome
        ):
            parent_connect_genes = parent_genome.get_connect_genes()
            child_nodes.update(
                find_required_nodes(connection_keys, parent_genome)
            )  # TODO: Is it right to also replace bias, not just connections?
            for connection in connection_keys:
                child_connections[connection] = parent_connect_genes[connection]
            return deepcopy(child_nodes), deepcopy(child_connections)

        p1_differences = p1_connections - p2_connections
        p2_differences = p2_connections - p1_connections
        if not draw:
            child_nodes, child_connections = _add_node_connections(
                child_nodes, child_connections, p1_differences, self.genome
            )
        else:
            for connection in p1_differences:
                if random.random() < 0.5:
                    child_nodes, child_connections = _add_node_connections(
                        child_nodes, child_connections, [connection], self.genome
                    )
            for connection in p2_differences:
                if random.random() < 0.5:
                    child_nodes, child_connections = _add_node_connections(
                        child_nodes, child_connections, [connection], spouse.genome
                    )
        child.replace_genome(child_nodes, child_connections)
        assert child._check_genome_model_synced()
        return child


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
            raise RuntimeError(
                "Expected {0:n} inputs, got {1:n}".format(
                    len(self.input_nodes), len(inputs)
                )
            )

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
        sensor_nodes = genome.get_node_keys("sensor")
        output_nodes = genome.get_node_keys("output")
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
