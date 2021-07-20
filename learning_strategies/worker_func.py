from mpi4py import MPI
import numpy as np

# MPI setting.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def run_rollout(env, network):
    count = 0
    while True:
        args = comm.recv(source=0, tag=1000)
        if args == "terminate":
            break
        off_id, offspring, eval_ep_num = args
        count += 1
        total_reward = 0
        for i in range(eval_ep_num):
            states = env.reset()
            done = False
            for k, model in offspring.items():
                model.reset()
            while not done:
                actions = {}
                for k, model in offspring.items():
                    s = states[k]["state"][np.newaxis, ...]
                    actions[k] = model(s)
                states, r, done, _ = env.step(actions)
                # env.render()
                total_reward += r
        rewards = total_reward / eval_ep_num
        comm.isend([rank, off_id, rewards], dest=0, tag=1001)
    print(f"successfully terminate rank {rank}")
