import os
import time
from datetime import datetime
from collections import deque
from mpi4py import MPI
import numpy as np
import torch
import wandb
from .abstracts import BaseESLoop
from collections import deque

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


class ESLoop(BaseESLoop):
    def __init__(
        self,
        config,
        offspring_strategy,
        agent_ids,
        env_name,
        network,
        generation_num,
        n_workers,
        eval_ep_num,
        log=False,
        save_model_period=10,
    ):
        super().__init__()
        self.network = network
        self.n_workers = n_workers
        self.offspring_strategy = offspring_strategy
        self.generation_num = generation_num
        self.eval_ep_num = eval_ep_num
        self.log = log
        self.save_model_period = save_model_period
        self.agent_ids = agent_ids

        self.network.zero_init()
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
            wandb.init(project=env_name, config=config)

    def reset_worker_status(self):
        self.free_worker = np.zeros(self.n_workers)

    def get_free_worker_rank(self):
        indices = np.argwhere(self.free_worker == 0)
        if len(indices) == 0:
            return None
        else:
            idx = indices[0][0]
            self.free_worker[idx] = 1
            return idx + 1

    def terminate_all_workers(self):
        for i in range(1, self.n_workers + 1):
            comm.send("terminate", dest=i, tag=1000)

    def run(self):

        # init offsprings
        offsprings = self.offspring_strategy.init_offspring(self.network, self.agent_ids)
        ep_num = 0
        for _ in range(self.generation_num):
            start_time = time.time()
            ep_num += 1

            arguments = [(off_id, off, self.eval_ep_num) for off_id, off in enumerate(offsprings)]
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
            offsprings, best_reward, curr_sigma = self.offspring_strategy.evaluate(rewards)
            eval_consumed_time = time.time() - eval_start_time

            # print log
            consumed_time = time.time() - start_time
            print(
                f"episode: {ep_num}, Best reward: {best_reward:.2f}, sigma: {curr_sigma:.3f}, time: {consumed_time:.2f}, rollout_t: {rollout_consumed_time:.2f}, eval_t: {eval_consumed_time:.2f}"
            )

            prev_reward = best_reward
            if self.log:
                self.ep5_rewards.append(best_reward)
                ep5_mean_reward = sum(self.ep5_rewards) / len(self.ep5_rewards)
                wandb.log({"ep5_mean_reward": ep5_mean_reward, "curr_sigma": curr_sigma})

            elite = self.offspring_strategy.get_elite_model()
            if ep_num % self.save_model_period == 0:
                save_pth = self.save_dir + "/saved_models" + f"/ep_{ep_num}.pt"
                torch.save(elite.state_dict(), save_pth)
        self.terminate_all_workers()