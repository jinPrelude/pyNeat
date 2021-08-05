def find_required_nodes(connections, genome):
    assert type(connections) in [list, set]
    network_nodes = genome.get_nodes()
    node_nums = set()
    nodes = {}
    connect_genes = genome.get_connect_genes()
    for connection in connections:
        gene = connect_genes[connection]
        node_nums.add(gene.in_node_num)
        node_nums.add(gene.out_node_num)
    for node_num in node_nums:
        nodes[node_num] = network_nodes[node_num]
    return nodes
