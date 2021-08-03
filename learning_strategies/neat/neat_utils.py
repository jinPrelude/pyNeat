from copy import deepcopy
import random

import numpy as np
import sys


def crossover_offsprings(survival_ratio, sorted_parents, sorted_rewards, offspring_num):
    offsprings = []
    parent_num = round(survival_ratio * len(sorted_parents))
    assert parent_num >= 2
    parents = sorted_parents[:parent_num]
    rewards = sorted_rewards[:parent_num]
    p = np.arange(1, parent_num + 1)[::-1] / sum(range(parent_num + 1))
    for _ in range(offspring_num):
        p1_idx, p2_idx = np.random.choice(range(parent_num), 2, p=p, replace=False)
        p1, p2 = parents[p1_idx], parents[p2_idx]
        superior = "p1"
        if rewards[p1_idx] < rewards[p2_idx]:
            superior = "p2"
        elif rewards[p1_idx] == rewards[p2_idx]:
            superior = "draw"
        child = _crossover(p1, p2, superior)
        offsprings.append(child)

    return offsprings


# TODO: Put crossover function inside the NEAT Class
def _crossover(parent1, parent2, superior):
    assert superior in ["p1", "p2", "draw"]
    child = deepcopy(parent1)
    p1_connections = set(parent1.genome.get_connect_genes().keys())
    p2_connections = set(parent2.genome.get_connect_genes().keys())

    # matching genes crossover
    matching_connections = p1_connections & p2_connections
    child_nodes = find_required_nodes(matching_connections, parent1)
    child_connections = {}
    p1_connect_genes = parent1.genome.get_connect_genes()
    p2_connect_genes = parent2.genome.get_connect_genes()
    for connection in matching_connections:
        rand_num = random.random()
        if rand_num > 0.5:
            child_connections[connection] = p1_connect_genes[connection]
        else:
            child_connections[connection] = p2_connect_genes[connection]

    child.update_model(child_nodes, child_connections)
    return child


def mutate_offsprings(offsprings):
    for off in offsprings:
        off.mutate()
    return offsprings


def find_required_nodes(connections, network):
    assert type(connections) in [list, set]
    network_nodes = network.genome.get_nodes()
    node_nums = set()
    nodes = {}
    connect_genes = network.genome.get_connect_genes()
    for connection in connections:
        gene = connect_genes[connection]
        node_nums.add(gene.in_node_num)
        node_nums.add(gene.out_node_num)
    network
    for node_num in node_nums:
        nodes[node_num] = network_nodes[node_num]
    return nodes
