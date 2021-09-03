import random
from copy import deepcopy
from typing import Tuple

from learning_strategies.abstracts import BaseOffspringStrategy
from learning_strategies.general_utils import wrap_agentid
from .neat_utils import *


class Neat(BaseOffspringStrategy):
    """Neat algorithm.

    Attributes
    ----------
    offspring_num : int
        Number of the offsprings.
    crossover_ratio : float
        Ratio of offsprings through crossover.
    champions_num : int
        Number of champions. Champions are included in the next offspring without mutate.
    survival_ratio :
        Only agents with a survival rate remain to produce offspring.
    c1 : float
        Coeffucuent that controls the weight of the number of different genes.
    c3 : float
        Coeffucuent that controls the weight of the difference in weight mean.
    delta_threshold : float
        Delta threshold.
    crossover_num : int
        Number of offsprings through crossover.
    mutate_only_num : int
        Number off offsprings only through mutation.
    survival_num : int
        Only agents with a survival num remain to produce offspring.
    elite_model : BaseNeat
        The highest-scoring agent ever.
    self.offsprings : List[BaseNeat]
        Current offsprings.

    """

    def __init__(
        self,
        offspring_num: float,
        crossover_ratio: float = 0.75,
        champions_num: int = 5,
        survival_ratio: float = 0.2,
        c1: float = 1.0,
        c3: float = 0.4,
        delta_threshold: float = 3.0,
    ):
        """Neat init method.

        Parameters
        ----------
        offspring_num : float
            Number of the offsprings.
        crossover_ratio : float, optional
            Ratio of offsprings through crossover, by default 0.75
        champions_num : int, optional
            Number of champions. Champions are included in the next offspring without mutate, by default 5
        survival_ratio : float, optional
            Only agents with a survival rate remain to produce offspring, by default 0.2
        c1 : float, optional
            Coeffucuent that controls the weight of the number of different genes, by default 1.0
        c3 : float, optional
            Coeffucuent that controls the weight of the difference in weight mean, by default 0.4
        delta_threshold : float, optional
            Delta threshold, by default 3.0
        """
        super(Neat, self).__init__()
        self.offspring_num = offspring_num
        self.crossover_ratio = crossover_ratio
        self.champions_num = champions_num
        self.survival_ratio = survival_ratio
        self.c1 = c1
        self.c3 = c3
        self.delta_threshold = delta_threshold

        self.crossover_num = round(self.crossover_ratio * self.offspring_num)
        self.mutate_only_num = (
            self.offspring_num - self.crossover_num - self.champions_num
        )
        self.survival_num = round(self.survival_ratio * self.offspring_num)
        self.survival_num = max(self.survival_num, 2)

        self.elite_model = None
        self.offsprings = []

    def get_elite_model(self) -> BaseNeat:
        """Return the current elite model

        Returns
        -------
        BaseNeat
            Elite model.
        """
        return self.elite_model

    def init_offspring(self, network: BaseNeat, agent_ids: list) -> dict:
        """Initializes and returns the agent. This function is called only the first time.

        Parameters
        ----------
        network : BaseNeat
            Random initialize this network to generate the initial agent.
        agent_ids : list
            Id of the agents that should be created.

        Returns
        -------
        dict
            Dictionary with key is id and value is network.
        """
        self.agent_ids = agent_ids
        offspring_group = []
        for _ in range(self.offspring_num):
            offspring = deepcopy(network)
            offspring.init_genome()
            self.offsprings.append(offspring)
            offspring_group.append(wrap_agentid(agent_ids, offspring))

        return offspring_group

    def evaluate(self, rewards: List[float]) -> Tuple[dict, dict]:
        """Evaluate the offsprings and return the next offsprings, and evaluate informations.

        Parameters
        ----------
        rewards : List[float]
            Rewards of the current offsprings.

        Returns
        -------
        Tuple[dict, dict]
            offspring_group: Newly created offsprings.
            info: Evaluation information to be logged.
        """
        self.offsprings, rewards = sort_offsprings_rewards(self.offsprings, rewards)
        self.elite_model = deepcopy(self.offsprings[0])  # deepcopy is essential
        champions = deepcopy(
            self.offsprings[: self.champions_num]
        )  # deepcopy is essneital
        offsprings_mean_reward = mean(rewards)
        # # adjust fitness
        delta_dict, diversity_score = get_delta_dict(self.offsprings, self.c1, self.c3)
        adjusted_fitness, pass_score = calculate_adjusted_fitness(
            self.offsprings, rewards, self.delta_threshold, delta_dict
        )

        survivals, survivals_rewards = pick_by_pass_score(
            self.offsprings, adjusted_fitness, pass_score, self.survival_num
        )
        survivals, survivals_rewards = (
            survivals[: self.survival_num],
            survivals_rewards[: self.survival_num],
        )
        crossover = crossover_offsprings(
            survivals,
            survivals_rewards,
            self.crossover_num,
            delta_dict,
            self.delta_threshold,
        )
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
