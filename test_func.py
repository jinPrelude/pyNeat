from copy import deepcopy

from networks.neat.network import NeatNetwork


def _get_weights_bias_list(network):
    connections = network.genome.get_connect_genes()
    nodes = network.genome.get_nodes()
    net_weights, net_bias = [], []
    for connect, node in zip(connections.values(), nodes.values()):
        net_weights.append(connect.weight)
        net_bias.append(node.bias)
    return net_weights, net_bias


def _get_node_evals_weights(network):
    node_weights = {}
    node_bias = {}
    node_evals = network.model.node_evals
    for eval in node_evals:
        curr_num = eval[0]
        node_bias[curr_num] = eval[2]
        input_list = eval[3]
        for input_num, weight in input_list:
            node_weights[(curr_num, input_num)] = weight
    return node_weights, node_bias


def test_normal_init():
    test_net = NeatNetwork(2, 1, False, 0, 1, 1, 30, -30, {})
    test_net.init_genome()

    # test expected nodes and connections
    expected_node_keys = [0, 1, 2]
    nodes = test_net.genome.get_nodes()
    output_node_keys = list(nodes.keys())
    assert expected_node_keys == output_node_keys

    expected_connection_keys = [(0, 2), (1, 2)]
    connections = test_net.genome.get_connect_genes()
    output_connection_keys = list(connections.keys())
    assert expected_connection_keys == output_connection_keys

    # test whether init_genome actually changes genome weights & bias
    before_weights, before_bias = _get_weights_bias_list(test_net)
    test_net.genome.normal_init(0, 1)
    after_weights, after_bias = _get_weights_bias_list(test_net)
    assert before_weights != after_weights
    assert before_bias != after_bias


def test_mutate_weight():
    test_net = NeatNetwork(2, 1, False, 0, 1, 1, 30, -30, {})
    test_net.init_genome()
    original_weights, _ = _get_weights_bias_list(test_net)
    test_net.genome.mutate_weight(1)  # mutate all weights

    # check if weights changed
    after_weights, _ = _get_weights_bias_list(test_net)
    assert original_weights != after_weights


def test_mutate_bias():
    test_net = NeatNetwork(2, 1, False, 0, 1, 1, 30, -30, {})
    test_net.init_genome()
    _, original_bias = _get_weights_bias_list(test_net)
    test_net.genome.mutate_bias(1)  # mutate all weights

    # check if weights changed
    _, after_bias = _get_weights_bias_list(test_net)
    assert original_bias != after_bias


def test_mutate_add_node():
    test_net = NeatNetwork(2, 1, False, 0, 1, 1, 30, -30, {})
    test_net.init_genome()
    original_nodes = deepcopy(test_net.genome.get_nodes())
    original_nodes_keys = set(original_nodes.keys())
    original_connections = deepcopy(test_net.genome.get_connect_genes())

    # check if new node is made and counted properly
    expected_new_node_num = 3
    test_net.genome.mutate_add_node(1)
    changed_nodes = test_net.genome.get_nodes()
    changed_nodes_keys = set(changed_nodes.keys())
    added_keys = list(changed_nodes_keys - original_nodes_keys)
    assert len(added_keys) == 1 and added_keys[0] == expected_new_node_num

    # check if existed connection is disabled
    new_connections = test_net.genome.get_connect_genes()
    input_node, output_node = None, None
    for connection in new_connections.keys():
        if connection[1] == expected_new_node_num:
            input_node = connection[0]
        if connection[0] == expected_new_node_num:
            output_node = connection[1]
    assert original_connections[(input_node, output_node)].enabled
    assert not new_connections[(input_node, output_node)].enabled


def test_mutate_add_connection():
    test_net = NeatNetwork(2, 1, False, 0, 1, 1, 30, -30, {})
    test_net.init_genome()
    original_connections = deepcopy(test_net.genome.get_connect_genes())
    original_connections_keys = set(original_connections.keys())

    # check if new connection is not made when hidden node is not exist.
    test_net.genome.mutate_add_connection(1)
    changed_connections = deepcopy(test_net.genome.get_connect_genes())
    changed_connections_keys = set(changed_connections.keys())
    assert original_connections_keys == changed_connections_keys

    # check if new connection is made after generate new nodes
    test_net.genome.node_genes.add_node("hidden")
    test_net.genome.mutate_add_connection(1)
    changed_connections = deepcopy(test_net.genome.get_connect_genes())
    changed_connections_keys = set(changed_connections.keys())
    new_added_connection_keys = changed_connections_keys - original_connections_keys
    assert len(new_added_connection_keys) == 1


if __name__ == "__main__":
    # NeatNetwork test
    # test_update_model() # TODO
    # test_crossover() # TODO

    # Genome test
    test_normal_init()
    test_mutate_weight()
    test_mutate_bias()  # TODO: mutate_bias is not implemented
    test_mutate_add_node()
    test_mutate_add_connection()
    # test_get_node_keys() # TODO
