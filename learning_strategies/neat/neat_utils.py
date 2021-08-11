from itertools import combinations

import numpy as np


def mutate_offsprings(offsprings):
    for off in offsprings:
        off.mutate()
    return offsprings


def crossover_offsprings(parents, rewards, offspring_num, delta_dict, delta_threshold):
    offsprings = []
    parents, rewards = sort_offsprings_rewards(parents, rewards)
    p = np.arange(1, len(parents) + 1)[::-1] / sum(range(len(parents) + 1))
    while len(offsprings) < offspring_num:
        p1_idx, p2_idx = np.random.choice(range(len(parents)), 2, p=p, replace=False)
        if delta_dict[(p1_idx, p2_idx)] > delta_threshold:
            continue
        p1, p2 = parents[p1_idx], parents[p2_idx]
        if rewards[p1_idx] < rewards[p2_idx]:
            child = p1.crossover(p2)
        elif rewards[p1_idx] > rewards[p2_idx]:
            child = p2.crossover(p1)
        else:
            child = p1.crossover(p2, draw=True)
        offsprings.append(child)

    return offsprings


def calculate_adjusted_fitness(offsprings, rewards, delta_threshold, delta_dict):
    # regularize rewards
    max_r = max(rewards)
    min_r = min(rewards)
    new_rewards = []
    for i in range(len(rewards)):
        new_rewards.append((rewards[i] - min_r) / (max_r - min_r))
    rewards = new_rewards

    # calculate adjusted fitness
    adjusted_fitness = []
    pass_score = []
    for i in range(len(offsprings)):
        same_speices_fitnesses = []
        fitness = rewards[i]
        same_speices_num = 0
        for j in range(len(offsprings)):
            if i == j:
                continue
            elif delta_dict[(i, j)] < delta_threshold:
                same_speices_fitnesses.append(rewards[j])
                same_speices_num += 1
        new_fitness = fitness / same_speices_num
        adjusted_fitness.append(new_fitness)
        pass_score.append(fitness - mean(same_speices_fitnesses))

    return adjusted_fitness, pass_score


def get_delta_dict(offsprings, c1, c3):
    delta_list = {}
    diversity_score = 0
    parent_indices = [i for i in range(len(offsprings))]
    parent_combs = list(combinations(parent_indices, 2))
    for (i, j) in parent_combs:
        delta_list[(i, j)] = calculate_delta(offsprings[i], offsprings[j], c1, c3)
        diversity_score += delta_list[(i, j)] / 1e6
        delta_list[(j, i)] = delta_list[(i, j)]
    return delta_list, diversity_score


def calculate_delta(p1, p2, c1, c3):
    p1_genes = p1.genome.get_connect_genes()
    p2_genes = p2.genome.get_connect_genes()
    p1_connect_keys = set(p1_genes.keys())
    p2_connect_keys = set(p2_genes.keys())
    all_genes_num = len(p1_connect_keys & p2_connect_keys)
    diff_genes_num = len(set.symmetric_difference(p1_connect_keys, p2_connect_keys))
    weight_diff = 1
    if (len(p1_genes) + len(p2_genes)) / 2 > 20:
        p1_weight_avg = get_weights_average(p1)
        p2_weight_avg = get_weights_average(p2)
        weight_diff = abs(p1_weight_avg - p2_weight_avg)
    delta = (c1 * diff_genes_num) / all_genes_num
    delta += c3 * weight_diff
    return delta


def get_weights_average(neat_network):
    # for delta calculation
    connect_genes = neat_network.genome.get_connect_genes()
    weights = []
    for gene in connect_genes.values():
        weights.append(gene.weight)
    return mean(weights)


def pick_by_pass_score(offsprings, adjusted_fitness, pass_scores, survival_num):
    # pick offsprings which fitness score is higher than speices' average.
    pass_num = len([i for i in pass_scores if i > 0])
    if pass_num < survival_num:
        # sort the offsprings by pass_scores if pass_num is insufficient
        rank_id = np.flip(np.argsort(pass_scores))
        survivals = [offsprings[i] for i in rank_id]
        survivals_rewards = [adjusted_fitness[i] for i in rank_id]
    else:
        survivals = []
        survivals_rewards = []
        for i, pass_score in enumerate(pass_scores):
            if pass_score > 0:
                survivals.append(offsprings[i])
                survivals_rewards.append(adjusted_fitness[i])

    return survivals, survivals_rewards


def sort_offsprings_rewards(offsprings, rewards):
    rank_id = np.flip(np.argsort(rewards))
    sorted_offsprings = [offsprings[i] for i in rank_id]
    sorted_rewards = [rewards[i] for i in rank_id]
    return sorted_offsprings, sorted_rewards


def mean(x_list):
    if len(x_list) == 0:
        return 0
    else:
        return sum(x_list) / len(x_list)
