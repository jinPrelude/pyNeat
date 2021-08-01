import numpy as np

##### Genome #####


class Genome:
    def __init__(self, num_state, num_action, mu=0.0, std=1.0):
        self.node_genes = NodeGenes(num_state, num_action)
        self.connect_genes = ConnectGenes()
        # initialize connect genes
        sensor_nodes = self.node_genes.get_sensor_nodes()
        output_nodes = self.node_genes.get_output_nodes()
        self.connect_genes.init_connection(sensor_nodes, output_nodes, mu, std)

        self.connections = self.connect_genes.genes_by_connect
        self.nodes = self.node_genes.nodes


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

    def get_sensor_nodes(self):
        return self.sensor_node_num_list

    def get_output_nodes(self):
        return self.output_node_num_list


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
        self.get_innov_num = iter(range(100000000))
        self.genes_by_innov = {}
        self.genes_by_connect = {}

    def init_connection(self, sensor_nodes: list, output_nodes: list, mu=0.0, std=1.0):
        for sensor_n in sensor_nodes:
            for output_n in output_nodes:
                weight = np.random.normal(mu, std)
                self.add_connection(sensor_n, output_n, weight, True)

    def add_connection(self, in_node_num, out_node_num, weight, enabled) -> int:
        innov_num = next(self.get_innov_num)
        self.genes_by_innov[innov_num] = Connect(in_node_num, out_node_num, weight, enabled, innov_num)
        self.genes_by_connect[(in_node_num, out_node_num)] = self.genes_by_innov[innov_num]
        return innov_num

    def get_connection_tuple_list(self):
        return list(self.genes_by_connect.keys())


class Connect:
    def __init__(self, in_node_num, out_node_num, weight, enabled, innov_num) -> None:
        self.connection = (in_node_num, out_node_num)
        self.in_node_num = in_node_num
        self.out_node_num = out_node_num
        self.weight = weight
        self.enabled = enabled
        self.innov_num = innov_num
