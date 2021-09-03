import random
from itertools import product
from typing import Dict, Tuple, List

import numpy as np


###### Node #######


class Node:
    """Node class that holds it's index, type, and bias.

    Attributes
    ----------
    index : int
        Index of the node.
    type : str
        Type of the node[sensor, output, hidden].
    bias : float
        Bias of the node.
    """

    def __init__(self, node_index: int, node_type: str, bias: float) -> None:
        """Node Init method.

        Parameters
        ----------
        node_index : int
            Index of the node.
        node_type : str
            Type of the node[sensor, output, hidden].
        bias : float
            Bias of the node.
        """
        assert node_type in ["sensor", "output", "hidden"]
        self.index = node_index
        self.type = node_type
        self.bias = bias


class NodeGenes:
    """Class that holds Node genes.

    Attributes
    ----------
    init_mu : float
        Initial value of the mu when bias are initialized using Gaussian noise.
    init_std : float
        Initial value of the std when bias are initialized using Gaussian noise.
    new_node_idx : int
        Index the node that will be created next.
    nodes : Dict[int, Node]
        Dictionary which the key is Index and value is Node.
    node_list_by_type : Dict[str, List[int]]
        Dictionary which the key is node type and value is list of indices of corresponding nodes.

    """

    def __init__(
        self, num_state: int, num_action: int, init_mu: float, init_std: float
    ) -> None:
        """NodeGenes Init method.

        Parameters
        ----------
        num_state : int
            State space.
        num_action : int
            Action space.
        init_mu : float
        Initial value of the mu when bias are initialized using Gaussian noise.
        init_std : float
            Initial value of the std when bias are initialized using Gaussian noise.
        """
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
        self, node_type: str, node_idx: int = None, bias: float = None
    ) -> int:
        """Add or overwrite the node using new informations.
        If the node_idx is exist, overwrite. Otherwise create new node.

        Parameters
        ----------
        node_type : str
            Type of the node[sensor, output, hidden].
        node_idx : int, optional
            Index of the node. If None create new node, by default None
        bias : float, optional
            bias of the node. If None normal initialize, by default None

        Returns
        -------
        int
            Index of newly created node.
        """
        # add node. if node_idx already exist overwrite.
        if node_idx is None:
            node_idx = self.new_node_idx
        if bias is None:
            bias = np.random.normal(self.init_mu, self.init_std)
        self.nodes[node_idx] = Node(node_idx, node_type, bias)
        self.node_list_by_type[node_type].append(node_idx)

        self.new_node_idx = max(self.new_node_idx, node_idx + 1)
        return node_idx

    def replace(self, nodes: Dict[int, Node]):
        """Replace ALL existing Node to new one.

        Parameters
        ----------
        nodes : Dict[int, Node]
            New node genes.
        """
        assert type(nodes) == dict
        assert type(nodes[0]) == Node
        self.nodes = nodes

    def get_keys_by_type(self, node_type: str) -> List[int]:
        """Return the list of specific type of node indices.

        Parameters
        ----------
        node_type : str
            Type of the node[sensor, output, hidden].

        Returns
        -------
        List[int]
            List of specific type of node indices.
        """
        assert node_type in [None, "all", "sensor", "output", "hidden"]
        if node_type in [None, "all"]:
            return list(self.nodes.keys())
        else:
            return list(x.index for x in self.nodes.values() if x.type == node_type)


###### Connect #######


class Connect:
    def __init__(self, in_node_num, out_node_num, weight, enabled) -> None:
        self.connection = (in_node_num, out_node_num)
        self.in_node_num = in_node_num
        self.out_node_num = out_node_num
        self.weight = weight
        self.enabled = enabled


class ConnectGenes:
    """Class that holds Connect genes.

    Attributes
    ----------
    connections: Dict[Tuple[int, int], Connect]
        Dictionary which the key is connection tuple and the value is Connect.
    init_mu : float
        Initial value of the mu when bias are initialized using Gaussian noise.
    init_std : float
        Initial value of the std when bias are initialized using Gaussian noise.

    """

    def __init__(self, init_mu: float, init_std: float) -> None:
        """Connect Genes init method. This method does not initialize the connections .
        To initialize the connections you should call init_connection().

        Parameters
        ----------
        init_mu : float
            Initial value of the mu when bias are initialized using Gaussian noise.
        init_std : float
            Initial value of the std when bias are initialized using Gaussian noise.
        """
        self.connections = {}
        self.init_mu = init_mu
        self.init_std = init_std

    def init_connection(self, sensor_nodes: List[int], output_nodes: List[int]):
        """Initialize connections using normal distribution.

        Parameters
        ----------
        sensor_nodes : List[int]
            List of sensor node indices.
        output_nodes : List[int]
            List of output node indices.
        """
        for sensor_n in sensor_nodes:
            for output_n in output_nodes:
                self.add_connection(sensor_n, output_n)

    def add_connection(
        self,
        in_node_index: int,
        out_node_index: int,
        weight: float = None,
        enabled: bool = True,
    ) -> None:
        """Add new connection

        Parameters
        ----------
        in_node_index : int
            In node index.
        out_node_index : int
            Out node index.
        weight : float, optional
            Weight of the node, by default None
        enabled : bool, optional
            Whether to enable the connection, by default True
        """
        if weight is None:
            weight = np.random.normal(self.init_mu, self.init_std)
        self.connections[(in_node_index, out_node_index)] = Connect(
            in_node_index, out_node_index, weight, enabled
        )

    def replace(self, connections: Dict[Tuple[int, int], Connect]):
        """Replace ALL existing connect genes to new one.

        Parameters
        ----------
        connections : Dict[Tuple[int, int], Connect
            New connect genes.
        """
        self.connections = connections


##### Genome #####
class Genome:
    """Class that holds node genes and connect genes.

    Attributes
    ----------
    init_mu : float
        Initial value of the mu when bias and weights are initialized using Gaussian noise.
    init_std : float
        Initial value of the std when bias and weights are initialized using Gaussian noise.
    mutate_std : float
        Std when new bias or weight are assigned by mutation using Gaussian noise.
    max_weight : float
        Maximum value that bias or weight can be assigned.
    min_weight : float
        Minimum value that bias or weight can be assigned.
    node_genes : NodeGenes
        Node genes.
    connect_genes : ConnectGenes
        Connect genes.
    """

    def __init__(
        self,
        num_state: int,
        num_action: int,
        init_mu: float,
        init_std: float,
        mutate_std: float,
        max_weight: float,
        min_weight: float,
    ):
        """Genome Init method.

        Parameters
        ----------
        num_state : int
            State space.
        num_action : int
            Action space.
        init_mu : float
            Initial value of the mu when bias and weights are initialized using Gaussian noise.
        init_std : float
            Initial value of the std when bias and weights are initialized using Gaussian noise.
        mutate_std : float
            Std when new bias or weight are assigned by mutation using Gaussian noise.
        max_weight : float
            Maximum value that bias or weight can be assigned.
        min_weight : float
            Minimum value that bias or weight can be assigned.
        """
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

    def normal_init(self, mu: float = None, std: float = None):
        """Normal initialize the weights and bias.

        Parameters
        ----------
        mu : float, optional
            mu of the normal distribution, by default None
        std : float, optional
            std if the normal distribution, by default None
        """
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

    def replace_genome(self, nodes: dict, connections: dict):
        """Replace the current node and connect genes with the new one.

        Parameters
        ----------
        nodes : dict
            New node genes.
        connections : dict
            New connect genes.
        """
        self.node_genes.replace(nodes)
        self.connect_genes.replace(connections)

    def get_nodes(self) -> Dict[int, Node]:
        """Return the node genes

        Returns
        -------
        dict
            Node genes where the key is node id and value is Node.
        """
        return self.node_genes.nodes

    def get_connect_genes(self) -> Dict[Tuple[int, int], Connect]:
        """Return the connect genes

        Returns
        -------
        Dict[Tuple[int, int], Connect]
            Connect genes where the key is connection tuple and value is Connect.
        """
        return self.connect_genes.connections

    def get_node_keys(self, node_type: str = None) -> List[int]:
        """Return the list of specific type of node indices.

        Parameters
        ----------
        node_type : str
            Type of the node[sensor, output, hidden].

        Returns
        -------
        List[int]
            List of specific type of node indices.
        """
        return self.node_genes.get_keys_by_type(node_type)

    def mutate_weight(self, prob: float) -> None:
        """Stochastic mutations of weights for all connect genes.

        Parameters
        ----------
        prob : float
            Probability.
        """
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

    def mutate_bias(self, prob: float) -> None:
        """Stochastic mutations of bias for all node genes.

        Parameters
        ----------
        prob : float
            Probability.
        """
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

    def mutate_add_node(self, prob: float) -> None:
        """Stochastically create new node and node gene.

        Parameters
        ----------
        prob : float
            Probability.
        """
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

    def mutate_add_connection(self, prob: float) -> None:
        """Stochastically create new random connection and connect gene.

        Parameters
        ----------
        prob : float
            Probability.
        """
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
