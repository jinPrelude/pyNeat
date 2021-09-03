from itertools import combinations
from typing import List, Dict, Tuple

import numpy as np

from networks.neat.abstracts import BaseNeat


def mutate_offsprings(offsprings: List[BaseNeat]) -> List[BaseNeat]:
    """Mutate all offsprings in the list.

    Parameters
    ----------
    offsprings : List[BaseNeat]
        Offsprings to be mutated.

    Returns
    -------
    List[BaseNeat]
        Mutataed offsprings.
    """
    for off in offsprings:
        off.mutate()
    return offsprings


def crossover_offsprings(
    survivals: List[BaseNeat],
    rewards: List[float],
    offspring_num: int,
    delta_dict: Dict[tuple, float],
    delta_threshold: float,
) -> List[BaseNeat]:
    """Crossover survivals and make new offsprings.

    Parameters
    ----------
    survivals : List[BaseNeat]
        Survivals of the previous rollout.
    rewards : List[float]
        Rewards of the survivals.
    offspring_num : int
        Number of the offsprings to be generated.
    delta_dict : Dict[tuple, float]
        Delta score of every survivals' combination.
    delta_threshold : float
        Delta threshold.

    Returns
    -------
    List[BaseNeat]
        Offsprings.
    """
    # calculate selection priority.
    rank_id = np.flip(np.argsort(rewards))
    prob_weight = np.arange(1, len(survivals) + 1)[::-1] / sum(
        range(len(survivals) + 1)
    )
    p = [prob_weight[x] for x in rank_id]

    offsprings = []
    while len(offsprings) < offspring_num:
        p1_idx, p2_idx = np.random.choice(range(len(survivals)), 2, p=p, replace=False)
        if delta_dict[(p1_idx, p2_idx)] > delta_threshold:
            continue
        p1, p2 = survivals[p1_idx], survivals[p2_idx]
        if rewards[p1_idx] < rewards[p2_idx]:
            child = p1.crossover(p2)
        elif rewards[p1_idx] > rewards[p2_idx]:
            child = p2.crossover(p1)
        else:
            child = p1.crossover(p2, draw=True)
        offsprings.append(child)

    return offsprings


def calculate_adjusted_fitness(
    offsprings: List[BaseNeat],
    rewards: List[float],
    delta_threshold: float,
    delta_dict: Dict[tuple, float],
) -> Tuple[List[float], List[float]]:
    """Calculate adjusted fitness of the offsprings.

    Parameters
    ----------
    offsprings : List[BaseNeat]
        Offsprings.
    rewards : List[float]
        Rewards of the offsprings.
    delta_threshold : float
        Delta threshold.
    delta_dict : Dict[tuple, float]
        Delta scores of every offsprings' combination.

    Returns
    -------
    Tuple[List[float], List[float]]
        adjusted fitness: adjusted fitness of the offsprings.
        pass_score: Difference between one's score and the same species' avg score.
                    Only the one with a positive pass_score survive.
    """
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


def get_delta_dict(
    offsprings: List[BaseNeat], c1: float, c3: float
) -> Tuple[dict, float]:
    """Calculate delta score of every offsprings' combination.

    Parameters
    ----------
    offsprings : List[BaseNeat]
        Offsprings.
    c1 : float
        Coeffucuent that controls the weight of the number of different genes.
    c3 : float
        Coeffucuent that controls the weight of the difference in weight mean.

    Returns
    -------
    Tuple[dict, float]
        delta_dict: Dict[Tuple[int, int], float]
            Delta scores of every offsprings' combination.
        diversity_score: float
            Value proportional to the mean of every delta score.
    """
    delta_dict = {}
    diversity_score = 0
    parent_indices = [i for i in range(len(offsprings))]
    parent_combs = list(combinations(parent_indices, 2))
    for (i, j) in parent_combs:
        delta_dict[(i, j)] = calculate_delta(offsprings[i], offsprings[j], c1, c3)
        diversity_score += delta_dict[(i, j)] / 1e6
        delta_dict[(j, i)] = delta_dict[(i, j)]
    return delta_dict, diversity_score


def calculate_delta(p1: BaseNeat, p2: BaseNeat, c1: float, c3: float) -> float:
    """Calculate delta score of the two agent.

    Parameters
    ----------
    p1 : BaseNeat
        Agent 1.
    p2 : BaseNeat
        Agent 2.
    c1 : float
        Coeffucuent that controls the weight of the number of different genes.
    c3 : float
        Coeffucuent that controls the weight of the difference in weight mean.

    Returns
    -------
    float
        Delta score of the two agent.
    """
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


def get_weights_average(neat_network: BaseNeat) -> float:
    """Get Average value of all network's wieghts.

    Parameters
    ----------
    neat_network : BaseNeat
        Agent.

    Returns
    -------
    float
        Average of all network's weights.
    """
    # for delta calculation
    connect_genes = neat_network.genome.get_connect_genes()
    weights = []
    for gene in connect_genes.values():
        weights.append(gene.weight)
    return mean(weights)


def pick_by_pass_score(
    offsprings: List[BaseNeat],
    adjusted_fitness: List[float],
    pass_scores: List[float],
    survival_num: int,
) -> Tuple[List[BaseNeat], List[float]]:
    """Return the agent which pass_score is positive.

    If the number of agents which pass_score is positive is less then
    survival_num, returns agent as many as survival_num in order of pass_score.

    Parameters
    ----------
    offsprings : List[BaseNeat]
        Offsprings.
    adjusted_fitness : List[float]
        Adjusted fitnesses of the offsprings.
    pass_scores : List[float]
        Pass scores of each offsprings.
    survival_num : int
        Minimum number of agents to be survived.

    Returns
    -------
    Tuple[List[BaseNeat], List[float]]
        survivals: List[BaseNeat]
            Agents which pass_score is positive.
        survivals_rewards: List[float]:
            rewards of survived agents.
    """
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


def sort_offsprings_rewards(
    offsprings: List[BaseNeat], rewards: List[float]
) -> Tuple[List[BaseNeat], List[float]]:
    """Sort offspring and rewards list in order of higher rewards.

    Parameters
    ----------
    offsprings : List[BaseNeat]
        Offsprings.
    rewards : List[float]
        Rewards of the offsprings.

    Returns
    -------
    Tuple[List[BaseNeat], List[float]]
        sorted_offsprings: List[BaseNeat]
        sorted_rewards: List[float]
    """
    rank_id = np.flip(np.argsort(rewards))
    sorted_offsprings = [offsprings[i] for i in rank_id]
    sorted_rewards = [rewards[i] for i in rank_id]
    return sorted_offsprings, sorted_rewards


def mean(x_list: List) -> float:
    """Returns the average value of the list.

    Parameters
    ----------
    x_list : List

    Returns
    -------
    float
        Average value of the list.
    """
    if len(x_list) == 0:
        return 0
    else:
        return sum(x_list) / len(x_list)
