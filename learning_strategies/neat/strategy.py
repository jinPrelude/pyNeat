from copy import deepcopy

import torch
import numpy as np

from learning_strategies.abstracts import BaseOffspringStrategy
from learning_strategies.general_utils import wrap_agentid
from .neat_utils import *


class Neat(BaseOffspringStrategy):
    def __init__(
        self,
        init_sigma,
        sigma_decay,
        elite_num,
        offspring_num,
        crossover_offspring_ratio=0.75,
        champions_num=5,
        survival_ratio=0.2,
    ):
        super(Neat, self).__init__()
        self.offspring_num = offspring_num
        self.init_sigma = init_sigma
        self.sigma_decay = sigma_decay
        self.elite_num = elite_num
        self.curr_sigma = self.init_sigma
        self.crossover_offspring_ratio = crossover_offspring_ratio
        self.champions_num = champions_num
        self.survival_ratio = survival_ratio

        self.crossover_num = round(self.crossover_offspring_ratio * self.offspring_num)
        self.mutate_only_num = self.offspring_num - self.crossover_num - self.champions_num
        self.survival_num = round(self.survival_ratio * self.offspring_num)

        self.elite_model = None
        self.offsprings = []

    def _gen_offsprings(self, agent_ids, elite_models, elite_num, offspring_num, curr_sigma):
        pass

    def get_elite_model(self):
        return self.elite_model

    def init_offspring(self, network: torch.nn.Module, agent_ids: list):
        self.agent_ids = agent_ids
        network.init_genes()
        offspring_group = []
        for _ in range(self.offspring_num):
            offspring = deepcopy(network)
            offspring.normal_init(0.0, self.init_sigma)
            self.offsprings.append(offspring)
            offspring_group.append(wrap_agentid(agent_ids, offspring))

        return offspring_group

    def evaluate(self, rewards: list):
        best_reward = max(rewards)
        offspring_rank_id = np.flip(np.argsort(rewards))
        self.offsprings = [self.offsprings[i] for i in offspring_rank_id]
        rewards = [rewards[i] for i in offspring_rank_id]
        self.elite_model = deepcopy(self.offsprings[0])  # deepcopy is essential

        survival_num = max(self.survival_num, 2)
        survivals = deepcopy(self.offsprings[:survival_num])
        mutate_only = [random.choice(survivals) for _ in range(self.mutate_only_num)]
        champions = deepcopy(self.offsprings[: self.champions_num])  # deepcopy is essneital
        crossover = crossover_offsprings(survivals, rewards, self.crossover_num)
        crossover = mutate_offsprings(crossover)
        mutate_only = mutate_offsprings(survivals)
        self.offsprings = champions + crossover + mutate_only
        offspring_group = [wrap_agentid(self.agent_ids, off) for off in self.offsprings]
        return offspring_group, best_reward, 0

    def get_wandb_cfg(self):
        pass
