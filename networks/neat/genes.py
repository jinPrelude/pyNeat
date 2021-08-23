import random
from itertools import product

import numpy as np


##### Genome #####
class Genome:
    def __init__(
        self,
        num_state,
        num_action,
        init_mu,
        init_std,
        mutate_std,
        max_weight,
        min_weight,
        mu=0.0,
        std=1.0,
    ):
        self.init_mu = init_mu
        self.init_std = init_std
        self.mutate_std = mutate_std
        self.max_weight = max_weight
        self.min_weight = min_weight

        self.node_genes = NodeGenes(num_state, num_action, init_mu, init_std)
        self.connect_genes = ConnectGenes(init_mu, init_std)
        # initialize connect genes
        sensor_nodes = self.get_node_keys("sensor")
        output_nodes = self.get_node_keys("output")
        self.connect_genes.init_connection(sensor_nodes, output_nodes)

    def normal_init(self, mu=None, std=None):
        if mu is None:
            mu = self.init_mu
        if std is None:
            std = self.init_std
        connect_genes = self.get_connect_genes()
        node_genes = self.get_nodes()
        for gene in connect_genes.values():
            gene.weight = np.random.normal(mu, std)
        for node in node_genes.values():
            node.bias = np.random.normal(mu, std)

    def replace_genome(self, nodes, connections):
        self.node_genes.replace(nodes)
        self.connect_genes.replace(connections)

    def get_nodes(self):
        return self.node_genes.nodes

    def get_connect_genes(self):
        return self.connect_genes.connections

    def get_node_keys(self, node_type=None):
        return self.node_genes.get_keys_by_type(node_type)

    def mutate_weight(self, prob):
        connect_genes = self.get_connect_genes()
        for gene in connect_genes.values():
            if random.random() < prob:
                if random.random() < 0.9:
                    # uniform perturb originally but I didn't understand how to implement it.
                    noise = np.random.normal(0, self.mutate_std)
                    weight = np.clip(
                        gene.weight + noise, self.min_weight, self.max_weight
                    )

                else:
                    weight = np.random.uniform(self.min_weight, self.max_weight)
                    weight = np.clip(weight, self.min_weight, self.max_weight)
                gene.weight = weight

    def mutate_bias(self, prob):
        nodes = self.get_nodes()
        for node in nodes.values():
            if random.random() < prob:
                if random.random() < 0.9:
                    # uniform perturb originally but I didn't understand how to implement it.
                    noise = np.random.normal(0, self.mutate_std)
                    bias = np.clip(node.bias + noise, self.min_weight, self.max_weight)

                else:
                    bias = np.random.uniform(self.min_weight, self.max_weight)
                    bias = np.clip(bias, self.min_weight, self.max_weight)
                node.bias = bias

    def mutate_add_node(self, prob):
        if random.random() < prob:
            connect_genes = self.get_connect_genes()
            conn_to_split = random.choice(list(connect_genes.values()))
            new_node_num = self.node_genes.add_overwrite_node("hidden")
            conn_to_split.enabled = False
            self.connect_genes.add_connection(
                conn_to_split.in_node_num, new_node_num, 1.0, True
            )
            self.connect_genes.add_connection(
                new_node_num, conn_to_split.out_node_num, 1.0, True
            )

    def mutate_add_connection(self, prob):
        if random.random() < prob:
            connections = self.get_connect_genes()
            connections_keys = set(connections.keys())
            output_node_keys = self.get_node_keys("output")
            hidden_node_keys = self.get_node_keys("hidden")
            output_node_candidates = output_node_keys + hidden_node_keys
            input_node_candidates = self.get_node_keys("all")
            possible_combs = set(product(input_node_candidates, output_node_candidates))
            # Don't allow connections between two output nodes
            possible_combs = set(
                x
                for x in possible_combs
                if not (x[0] in output_node_keys and x[1] in output_node_keys)
            )
            possible_combs = list(possible_combs - connections_keys)
            if len(possible_combs) == 0:
                return
            (input_node_num, output_node_num) = random.choice(possible_combs)
            if (input_node_num, output_node_num) in connections.keys():
                return
            self.connect_genes.add_connection(input_node_num, output_node_num)


###### Node #######
class NodeGenes:
    def __init__(self, num_state, num_action, init_mu, init_std) -> None:
        self.init_mu = init_mu
        self.init_std = init_std
        self.new_node_idx = 0
        self.nodes = {}
        self.node_list_by_type = {"sensor": [], "output": [], "hidden": []}
        for _ in range(num_state):
            self.add_overwrite_node("sensor")
        for _ in range(num_action):
            self.add_overwrite_node("output")

    def add_overwrite_node(
        self, node_type: str, node_num: int = None, bias=None
    ) -> int:
        # add node. if node_num already exist overwrite.
        if node_num is None:
            node_num = self.new_node_idx
        if bias is None:
            bias = np.random.normal(self.init_mu, self.init_std)
        self.nodes[node_num] = Node(node_num, node_type, bias)
        self.node_list_by_type[node_type].append(node_num)

        self.new_node_idx = max(self.new_node_idx, node_num + 1)
        return node_num

    def replace(self, nodes):
        assert type(nodes) == dict
        assert type(nodes[0]) == Node
        self.nodes = nodes

    def get_keys_by_type(self, node_type):
        assert node_type in [None, "all", "sensor", "output", "hidden"]
        if node_type in [None, "all"]:
            return list(self.nodes.keys())
        else:
            return list(x.num for x in self.nodes.values() if x.type == node_type)


class Node:
    def __init__(self, node_num, node_type, bias) -> None:
        assert node_type in ["sensor", "output", "hidden"]
        self.num = node_num
        self.type = node_type
        self.bias = bias


###### Connect #######
class ConnectGenes:
    def __init__(self, init_mu, init_std) -> None:
        self.connections = {}
        self.init_mu = init_mu
        self.init_std = init_std

    def init_connection(self, sensor_nodes: list, output_nodes: list):
        for sensor_n in sensor_nodes:
            for output_n in output_nodes:
                self.add_connection(sensor_n, output_n)

    def add_connection(
        self, in_node_num, out_node_num, weight=None, enabled=True
    ) -> int:
        if weight is None:
            weight = np.random.normal(self.init_mu, self.init_std)
        self.connections[(in_node_num, out_node_num)] = Connect(
            in_node_num, out_node_num, weight, enabled
        )

    def replace(self, connections):
        self.connections = connections


class Connect:
    def __init__(self, in_node_num, out_node_num, weight, enabled) -> None:
        self.connection = (in_node_num, out_node_num)
        self.in_node_num = in_node_num
        self.out_node_num = out_node_num
        self.weight = weight
        self.enabled = enabled
