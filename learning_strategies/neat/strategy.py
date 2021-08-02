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
        max_weight,
        min_weight,
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
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.elite_num = elite_num
        self.curr_sigma = self.init_sigma
        self.crossover_offspring_ratio = crossover_offspring_ratio
        self.champions_num = champions_num
        self.survival_ratio = survival_ratio

        self.innov_num_iterator = count()

        self.offsprings = []

    def _gen_offsprings(self, agent_ids, elite_models, elite_num, offspring_num, curr_sigma):
        pass

    def get_elite_model(self):
        pass

    def init_offspring(self, network: torch.nn.Module, agent_ids: list):
        self.agent_ids = agent_ids
        network.init_genes(self.innov_num_iterator)
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

        crossover_num = round(self.crossover_offspring_ratio * self.offspring_num)
        mutate_only_num = self.offspring_num - crossover_num

        champions = self.offsprings[: self.champions_num]
        # crossover = crossover_offsprings(self.survival_ratio, self.offsprings, rewards, crossover_num)
        # mutation = self.mutate(self.elite_num, self.offsprings, rewards, mutate_only_num)
        # self.offsprings = champions + crossover + mutation
        crossover = crossover_offsprings(self.survival_ratio, self.offsprings, rewards, self.offspring_num - self.champions_num)
        self.offsprings = crossover + champions
        offspring_group = [wrap_agentid(self.agent_ids, off) for off in self.offsprings]
        return offspring_group, best_reward, 0

    @staticmethod
    def mutate(elite_num, offsprings, rewards, offspring_num):
        pass

    def get_wandb_cfg(self):
        pass
