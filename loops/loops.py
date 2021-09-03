import os
from abc import *
import time
from datetime import datetime
from copy import deepcopy
from collections import deque
from mpi4py import MPI
import numpy as np
import wandb

from networks.neat.abstracts import BaseNeat
from learning_strategies.abstracts import BaseOffspringStrategy

# from moviepy.editor import ImageSequenceClip
# from pyvirtualdisplay import Display
# import builder

from .abstracts import BaseESLoop

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


class ESLoop(BaseESLoop):
    """Main class where evaluation loop is executed.

    Attributes
    ----------
    network: BaseNeat
        It is used to initialize agents in a strategy class.
    n_workers: int
        Number of process for rollout.
    offspring_strategy: BaseOffspringStrategy
        Offspring strategy.
    generation_num: int
        Determine how many rollouts to repeat.
    eval_ep_num: int
        Determine how many repeated evaluations will be made for each agent.
    log: bool
        Wheter you use wandb logging.
    save_model_period: int
        How often to leave wandb log and save model.
    agent_ids: list
        Agent IDs required for episode.
    env_name: str
        Name of the environment.
    self.save_dir: str
        Path to store the network for logging.

    """

    def __init__(
        self,
        config: dict,
        offspring_strategy: BaseOffspringStrategy,
        agent_ids: list,
        env_name: str,
        network: BaseNeat,
        generation_num: int,
        n_workers: int,
        eval_ep_num: int,
        log: bool = False,
        save_model_period: int = 10,
    ):
        """ESLoop init method.

        Parameters
        ----------
        config : dict
            Provides information for building environment and wandb logs.
        offspring_strategy : BaseOffspringStrategy
            Offspring strategy.
        agent_ids : list
            Agent IDs required for episode.
        env_name : str
            Name of the environment.
        network : BaseNeat
            It is used to initialize agents in a strategy class.
        generation_num : int
            Determine how many rollouts to repeat.
        n_workers : int
            Number of process for rollout.
        eval_ep_num : int
            Determine how many repeated evaluations will be made for each agent.
        log : bool, optional
            Wheter you use wandb logging, by default False
        save_model_period : int, optional
            How often to leave wandb log and save model, by default 10
        """
        super().__init__()
        self.network = network
        self.n_workers = n_workers
        self.offspring_strategy = offspring_strategy
        self.generation_num = generation_num
        self.eval_ep_num = eval_ep_num
        self.log = log
        self.save_model_period = save_model_period
        self.agent_ids = agent_ids
        self.env_name = env_name

        self.ep5_rewards = deque(maxlen=5)
        self.reset_worker_status()
        # create log directory
        now = datetime.now()
        curr_time = now.strftime("%Y%m%d%H%M%S")
        dir_lst = []
        self.save_dir = f"logs/{env_name}/{curr_time}"
        dir_lst.append(self.save_dir)
        dir_lst.append(self.save_dir + "/saved_models/")
        for _dir in dir_lst:
            os.makedirs(_dir)

        if self.log:
            # self.display = Display(visible=0, size=(300, 300))
            # self.display.start()
            wandb.init(project=env_name, config=config)
            self.env_cfg = config["env"]

    def reset_worker_status(self):
        """Reset all worker status to 0. Free workers are 0, busy workers are 1."""
        self.free_worker = np.zeros(self.n_workers)

    def get_free_worker_rank(self) -> int:
        """Randomly returns one of the id of the free worker.

        Returns
        -------
        int
            One of the free worker's ID.
        """
        indices = np.argwhere(self.free_worker == 0)
        if len(indices) == 0:
            return None
        else:
            idx = indices[0][0]
            self.free_worker[idx] = 1
            return idx + 1

    def terminate_all_workers(self):
        """Terminate all workers by sending "terminate" message."""
        for i in range(1, self.n_workers + 1):
            comm.send("terminate", dest=i, tag=1000)

    def run(self):
        """Start training."""
        # init offsprings
        offsprings = self.offspring_strategy.init_offspring(
            self.network, self.agent_ids
        )
        ep_num = 0
        for _ in range(self.generation_num):
            start_time = time.time()
            ep_num += 1

            arguments = [
                (off_id, off, self.eval_ep_num) for off_id, off in enumerate(offsprings)
            ]
            # rollout
            rollout_start_time = time.time()
            rewards = np.zeros(len(offsprings))
            remain_arg = deque(i for i in range(len(offsprings)))
            arg_idx = 0
            while len(remain_arg) != 0:
                id = self.get_free_worker_rank()
                if id is None or arg_idx >= len(offsprings):
                    req = comm.irecv(tag=1001)
                    data = req.wait()
                    finished_rank = data[0]
                    off_id = data[1]
                    rwd = data[2]
                    self.free_worker[finished_rank - 1] = 0
                    rewards[off_id] = rwd
                    remain_arg.pop()
                    continue
                comm.send(arguments[arg_idx], dest=id, tag=1000)
                arg_idx += 1
            self.reset_worker_status()

            rollout_consumed_time = time.time() - rollout_start_time

            eval_start_time = time.time()
            best_reward = max(rewards)
            offsprings, info = self.offspring_strategy.evaluate(rewards)
            eval_consumed_time = time.time() - eval_start_time

            # print log
            consumed_time = time.time() - start_time
            print(
                f"episode: {ep_num}, Best reward: {best_reward:.2f}, time: {consumed_time:.2f}, rollout_t: {rollout_consumed_time:.2f}, eval_t: {eval_consumed_time:.2f}"
            )

            if self.log:
                self.ep5_rewards.append(best_reward)
                info["ep5_mean_reward"] = sum(self.ep5_rewards) / len(self.ep5_rewards)
                wandb.log(info)

            elite = self.offspring_strategy.get_elite_model()
            if ep_num % self.save_model_period == 0:
                if self.log:
                    if "Unity" not in self.env_name:
                        # test_log_model(self.save_dir, self.env_cfg, elite, ep_num)
                        pass
                    save_pth = self.save_dir + "/saved_models/"
                    save_path_list = elite.save_model(save_pth, f"ep_{ep_num}")
                    for path in save_path_list:
                        wandb.save(path)
        self.terminate_all_workers()
        # self.display.stop()


# TODO: Disabled this function because it violates the
# Acyclic Dependecies Principle(ADP) of the dependency graph, where
# it import builder.

# def test_log_model(save_dir, env_cfg, elite_network, ep_num):
#     env = builder.build_env(env_cfg, rank)
#     agent_ids = env.get_agent_ids()

#     models = {}
#     for agent_id in agent_ids:
#         models[agent_id] = deepcopy(elite_network)
#         models[agent_id].reset()
#     obs = env.reset()

#     done = False
#     episode_reward = 0
#     ep_step = 0
#     ep_render_lst = []
#     while not done:
#         actions = {}
#         for k, model in models.items():
#             s = obs[k]["state"][np.newaxis, ...]
#             actions[k] = model.forward(s)
#         obs, r, done, _ = env.step(actions)
#         rgb_array = env.render(mode="rgb_array")
#         ep_render_lst.append(rgb_array)
#         episode_reward += r
#         ep_step += 1
#     clip = ImageSequenceClip(ep_render_lst[::2], fps=30)
#     clip.write_gif(os.path.join(save_dir, "play.gif"), fps=30)
#     wandb.save(os.path.join(save_dir, "play.gif"))

#     wandb.log({"test_reward": episode_reward})

#     del ep_render_lst

#     return episode_reward
