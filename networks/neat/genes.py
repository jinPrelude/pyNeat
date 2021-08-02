import numpy as np
import random
from copy import deepcopy

import numpy as np

##### Genome #####


class Genome:
    def __init__(self, num_state, num_action, mutate_sigma, max_weight, min_weight, innov_num_iterator, mu=0.0, std=1.0):
        self.mutate_sigma = mutate_sigma
        self.max_weight = max_weight
        self.min_weight = min_weight

        self.node_genes = NodeGenes(num_state, num_action)
        self.connect_genes = ConnectGenes()
        # initialize connect genes
        sensor_nodes = self.get_sensor_nodes()
        output_nodes = self.get_output_nodes()
        self.connect_genes.init_connection(sensor_nodes, output_nodes, innov_num_iterator, mu, std)

    def normal_init(self, mu, std):
        connect_genes = self.get_connect_genes()
        for gene in connect_genes.values():
            gene.weight = np.random.normal(mu, std)

    def update_genome(self, nodes, connections_by_innov):
        self.node_genes.update(nodes)
        self.connect_genes.update_by_innov(connections_by_innov)

    def get_nodes(self):
        return self.node_genes.nodes

    def get_sensor_nodes(self):
        return self.node_genes.sensor_node_num_list

    def get_output_nodes(self):
        return self.node_genes.output_node_num_list

    def get_connect_genes(self, key="innov_num"):
        assert key in ["innov_num", "connection"]
        if key == "innov_num":
            return self.connect_genes.genes_by_innov
        else:
            return self.connect_genes.genes_by_connect

    def get_innov_num_keys(self):
        return list(self.connect_genes.genes_by_innov.keys())

    def mutate_weight(self):
        connect_genes = self.get_connect_genes()
        for gene in connect_genes.values():
            r = random.random()
            if r < 0.8:
                r2 = random.random()
                if r2 < 0.9:
                    # uniform perturb originally but I didn't understand how to implement it.
                    gene.weight += np.random.normal(0, self.mutate_sigma)
                else:
                    gene.weight = np.random.uniform(self.min_weight, self.max_weight)


###### Node #######


class NodeGenes:
    def __init__(self, num_state, num_action) -> None:
        self.get_node_num = iter(range(100000000))
        self.nodes = {}
        self.sensor_node_num_list = []
        self.output_node_num_list = []
        for _ in range(num_state):
            node_num = self.add_node("sensor")
            self.sensor_node_num_list.append(node_num)
        for _ in range(num_action):
            node_num = self.add_node("output")
            self.output_node_num_list.append(node_num)

    def add_node(self, node_type: str, bias=None) -> int:
        node_num = next(self.get_node_num)
        self.nodes[node_num] = Node(node_num, node_type, bias)
        return node_num

    def update(self, nodes):
        assert type(nodes) == dict
        assert type(nodes[0]) == Node
        self.nodes = nodes
        self.sensor_node_num_list = []
        self.output_node_num_list = []
        for node in self.nodes.values():
            if node.type == "sensor":
                self.sensor_node_num_list.append(node.num)
            elif node.type == "output":
                self.output_node_num_list.append(node.num)


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
        self.genes_by_connect = {}
        self.genes_by_innov = {}

    def init_connection(self, sensor_nodes: list, output_nodes: list, innov_num_iterator, mu=0.0, std=1.0):
        for sensor_n in sensor_nodes:
            for output_n in output_nodes:
                weight = np.random.normal(mu, std)
                innov_num = next(innov_num_iterator)
                self.add_connection(sensor_n, output_n, weight, True, innov_num)

    def add_connection(self, in_node_num, out_node_num, weight, enabled, innov_num) -> int:
        self.genes_by_connect[(in_node_num, out_node_num)] = Connect(in_node_num, out_node_num, weight, enabled, innov_num)
        self.genes_by_innov[innov_num] = self.genes_by_connect[(in_node_num, out_node_num)]

    def update_by_innov(self, genes_by_innov):
        self.genes_by_innov = deepcopy(genes_by_innov)
        self.genes_by_connect = {}
        for gene in self.genes_by_innov.values():
            connection = gene.connection
            self.genes_by_connect[connection] = gene


class Connect:
    def __init__(self, in_node_num, out_node_num, weight, enabled, innov_num) -> None:
        self.connection = (in_node_num, out_node_num)
        self.in_node_num = in_node_num
        self.out_node_num = out_node_num
        self.weight = weight
        self.enabled = enabled
        self.innov_num = innov_num
