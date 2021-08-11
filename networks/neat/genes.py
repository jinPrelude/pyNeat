import numpy as np
import random

import numpy as np


##### Genome #####
class Genome:
    def __init__(self, num_state, num_action, mutate_sigma, max_weight, min_weight, mu=0.0, std=1.0):
        self.mutate_sigma = mutate_sigma
        self.max_weight = max_weight
        self.min_weight = min_weight

        self.node_genes = NodeGenes(num_state, num_action)
        self.connect_genes = ConnectGenes()
        # initialize connect genes
        sensor_nodes = self.get_node_keys("sensor")
        output_nodes = self.get_node_keys("output")
        self.connect_genes.init_connection(sensor_nodes, output_nodes, mu, std)

    def normal_init(self, mu, std):
        connect_genes = self.get_connect_genes()
        for gene in connect_genes.values():
            gene.weight = np.random.normal(mu, std)

    def replace_genome(self, nodes, connections):
        self.node_genes.replace(nodes)
        self.connect_genes.replace(connections)

    def get_nodes(self):
        return self.node_genes.nodes

    def get_connect_genes(self):
        return self.connect_genes.connections

    def get_node_keys(self, node_type=None):
        assert node_type in [None, "all", "sensor", "output", "hidden"]
        if node_type in [None, "all"]:
            return list(self.node_genes.nodes.keys())
        else:
            return self.node_genes.node_list_by_type[node_type]

    def mutate_weight(self, prob):
        connect_genes = self.get_connect_genes()
        for gene in connect_genes.values():
            if random.random() < prob:
                if random.random() < 0.9:
                    # uniform perturb originally but I didn't understand how to implement it.
                    noise = np.random.normal(0, self.mutate_sigma)
                    weight = np.clip(gene.weight + noise, self.min_weight, self.max_weight)

                else:
                    weight = np.random.uniform(self.min_weight, self.max_weight)
                    weight = np.clip(weight, self.min_weight, self.max_weight)
                gene.weight = weight

    def mutate_add_node(self, prob):
        if random.random() < prob:
            connect_genes = self.get_connect_genes()
            conn_to_split = random.choice(list(connect_genes.values()))
            new_node_num = self.node_genes.add_node("hidden")
            conn_to_split.enabled = False
            self.connect_genes.add_connection(conn_to_split.in_node_num, new_node_num, 1.0, True)
            self.connect_genes.add_connection(new_node_num, conn_to_split.out_node_num, 1.0, True)

    def mutate_add_connection(self, prob):
        if random.random() < prob:
            output_node_keys = self.get_node_keys("output")
            hidden_node_keys = self.get_node_keys("hidden")
            # print("mutate_add_connection: output_node_keys: ", output_node_keys)
            # print("mutate_add_connection: hidden_node_keys: ", hidden_node_keys)
            output_node_candidates = output_node_keys + hidden_node_keys
            output_node_num = random.choice(output_node_candidates)
            input_node_candidates = self.get_node_keys("all")
            # print("mutate_add_connection: all nodes: ", input_node_candidates)
            input_node_num = random.choice(input_node_candidates)
            connections = self.get_connect_genes()
            if (input_node_num, output_node_num) in connections.keys():
                return
            elif input_node_num in output_node_keys and output_node_num in output_node_keys:
                return
            self.connect_genes.add_connection(input_node_num, output_node_num)
            # print("mutate_add_connection: input_node: ", input_node_num, "\toutput: node: ", output_node_num)


###### Node #######
class NodeGenes:
    def __init__(self, num_state, num_action) -> None:
        self.get_node_num = iter(range(100000000))
        self.nodes = {}
        self.node_list_by_type = {"sensor": [], "output": [], "hidden": []}
        for _ in range(num_state):
            self.add_node("sensor")
        for _ in range(num_action):
            self.add_node("output")

    def add_node(self, node_type: str, bias=None) -> int:
        node_num = next(self.get_node_num)
        self.nodes[node_num] = Node(node_num, node_type, bias)
        self.node_list_by_type[node_type].append(node_num)
        return node_num

    def replace(self, nodes):
        assert type(nodes) == dict
        assert type(nodes[0]) == Node
        self.nodes = nodes
        self.node_list_by_type = {"sensor": [], "output": [], "hidden": []}
        for node in self.nodes.values():
            self.node_list_by_type[node.type].append(node.num)


class Node:
    def __init__(self, node_num, node_type, bias) -> None:
        assert node_type in ["sensor", "output", "hidden"]
        self.num = node_num
        self.type = node_type
        if bias is None:
            self.bias = np.random.normal(0, 1)
        else:
            self.bias = bias


###### Connect #######
class ConnectGenes:
    def __init__(self) -> None:
        self.connections = {}
        self.init_mu = None
        self.init_std = None

    def init_connection(self, sensor_nodes: list, output_nodes: list, mu=0.0, std=1.0):
        self.init_mu = mu
        self.init_std = std
        for sensor_n in sensor_nodes:
            for output_n in output_nodes:
                self.add_connection(sensor_n, output_n)

    def add_connection(self, in_node_num, out_node_num, weight=None, enabled=True) -> int:
        if weight is None:
            assert self.init_mu is not None, "init_connection() required to be called first."
            weight = np.random.normal(self.init_mu, self.init_std)
        self.connections[(in_node_num, out_node_num)] = Connect(in_node_num, out_node_num, weight, enabled)

    def replace(self, connections):
        self.connections = connections


class Connect:
    def __init__(self, in_node_num, out_node_num, weight, enabled) -> None:
        self.connection = (in_node_num, out_node_num)
        self.in_node_num = in_node_num
        self.out_node_num = out_node_num
        self.weight = weight
        self.enabled = enabled
