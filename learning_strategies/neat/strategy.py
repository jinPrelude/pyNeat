import random
from copy import deepcopy

import torch
import numpy as np

from learning_strategies.abstracts import BaseOffspringStrategy
from learning_strategies.general_utils import wrap_agentid
from .neat_utils import *


class Neat(BaseOffspringStrategy):
    def __init__(
        self,
        offspring_num,
        crossover_ratio=0.75,
        champions_num=5,
        survival_ratio=0.2,
        c1=1.0,
        c3=0.4,
        delta_threshold=3.0,
    ):
        super(Neat, self).__init__()
        self.offspring_num = offspring_num
        self.crossover_ratio = crossover_ratio
        self.champions_num = champions_num
        self.survival_ratio = survival_ratio
        self.c1 = c1
        self.c3 = c3
        self.delta_threshold = delta_threshold

        self.crossover_num = round(self.crossover_ratio * self.offspring_num)
        self.mutate_only_num = self.offspring_num - self.crossover_num - self.champions_num
        self.survival_num = round(self.survival_ratio * self.offspring_num)
        self.survival_num = max(self.survival_num, 2)

        self.elite_model = None
        self.offsprings = []

    def get_elite_model(self):
        return self.elite_model

    def init_offspring(self, network: torch.nn.Module, agent_ids: list):
        self.agent_ids = agent_ids
        offspring_group = []
        for _ in range(self.offspring_num):
            offspring = deepcopy(network)
            offspring.init_genome()
            self.offsprings.append(offspring)
            offspring_group.append(wrap_agentid(agent_ids, offspring))

        return offspring_group

    def evaluate(self, rewards: list):
        self.offsprings, rewards = sort_offsprings_rewards(self.offsprings, rewards)
        self.elite_model = deepcopy(self.offsprings[0])  # deepcopy is essential
        champions = deepcopy(self.offsprings[: self.champions_num])  # deepcopy is essneital
        offsprings_mean_reward = mean(rewards)
        # # adjust fitness
        delta_dict, diversity_score = get_delta_dict(self.offsprings, self.c1, self.c3)
        adjusted_fitness, pass_score = calculate_adjusted_fitness(self.offsprings, rewards, self.delta_threshold, delta_dict)

        survivals, survivals_rewards = pick_by_pass_score(self.offsprings, adjusted_fitness, pass_score, self.survival_num)
        survivals, survivals_rewards = survivals[: self.survival_num], survivals_rewards[: self.survival_num]
        crossover = crossover_offsprings(survivals, survivals_rewards, self.crossover_num, delta_dict, self.delta_threshold)
        crossover = mutate_offsprings(crossover)
        mutate_only = [random.choice(survivals) for _ in range(self.mutate_only_num)]
        mutate_only = mutate_offsprings(mutate_only)
        self.offsprings = champions + crossover + mutate_only
        offspring_group = [wrap_agentid(self.agent_ids, off) for off in self.offsprings]
        info = {
            "diversity_score": diversity_score,
            "offsprings_mean_reward": offsprings_mean_reward,
        }
        return offspring_group, info
