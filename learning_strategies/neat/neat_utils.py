from copy import deepcopy
import random

import numpy as np
import sys


def count(start=0, step=1):
    n = start
    while True:
        yield n
        n += step


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


def _crossover(parent1, parent2, superior):
    assert superior in ["p1", "p2", "draw"]
    child = deepcopy(parent1)
    p1_innov_nums = set(parent1.genome.get_innov_num_keys())
    p2_innov_nums = set(parent2.genome.get_innov_num_keys())

    # matching genes crossover
    matching_innov_nums = p1_innov_nums & p2_innov_nums
    child_nodes = find_required_nodes(matching_innov_nums, parent1)
    child_connections = {}
    p1_connect_genes = parent1.genome.get_connect_genes(key="innov_num")
    p2_connect_genes = parent2.genome.get_connect_genes(key="innov_num")
    for innov_num in matching_innov_nums:
        rand_num = random.random()
        if rand_num > 0.5:
            child_connections[innov_num] = p1_connect_genes[innov_num]
        else:
            child_connections[innov_num] = p2_connect_genes[innov_num]

    child.update_model(child_nodes, child_connections)
    return child


def find_required_nodes(innov_nums, network):
    assert type(innov_nums) in [list, set]
    network_nodes = network.genome.get_nodes()
    node_nums = set()
    nodes = {}
    connect_genes = network.genome.get_connect_genes(key="innov_num")
    for innov_num in innov_nums:
        gene = connect_genes[innov_num]
        node_nums.add(gene.in_node_num)
        node_nums.add(gene.out_node_num)
    network
    for node_num in node_nums:
        nodes[node_num] = network_nodes[node_num]
    return nodes
