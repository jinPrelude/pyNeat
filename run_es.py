import os
import sys
import signal
import subprocess
import argparse

import yaml
from mpi4py import MPI

import builder
from learning_strategies.worker_func import run_rollout

# keyboard interrupt(ctrl + c) handler.
def sigterm_handler(signal, frame):
    print("abort all processes")
    comm.Abort()
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, sigterm_handler)

# MPI setting.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def mpi_fork(n):
    """Re-launches the current script with workers
    Returns "parent" for original parent, "child" for MPI children
    (from https://github.com/garymcintire/mpi_util/)
    """
    if n <= 1:
        return "child"
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", IN_MPI="1")
        print(["mpiexec", "-n", str(n), sys.executable] + sys.argv)
        subprocess.check_call(["mpiexec", "-n", str(n), sys.executable] + ["-u"] + sys.argv, env=env)
        return "parent"
    else:
        global nworkers, rank
        nworkers = comm.Get_size()
        rank = comm.Get_rank()
        print("assigning the rank and nworkers", nworkers, rank)
        return "child"


def main(config, seed, n_workers, generation_num, eval_ep_num, log, save_model_period):
    env = builder.build_env(config["env"])
    agent_ids = env.get_agent_ids()
    env_name = env.name
    env.close()
    del env
    network = builder.build_network(config["network"])

    loop = builder.build_loop(
        config,
        network,
        agent_ids,
        env_name,
        generation_num,
        n_workers,
        eval_ep_num,
        log,
        save_model_period,
    )
    loop.run()


def worker(env_cfg, network_cfg):
    env = builder.build_env(env_cfg)
    network = builder.build_network(network_cfg)
    run_rollout(env, network)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, default="conf/cartpole_openaies.yaml", help="config file to run.")
    parser.add_argument("--seed", type=int, default=0, help="random seed.")
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--generation-num", type=int, default=10000, help="max number of generation iteration.")
    parser.add_argument("--eval-ep-num", type=int, default=5, help="number of model evaluaion per iteration.")
    parser.add_argument("--log", action="store_true", help="wandb log")
    parser.add_argument("--save-model-period", type=int, default=10, help="save model for every n iteration.")
    args = parser.parse_args()
    if "parent" == mpi_fork(args.n_workers + 1):
        print("abort all processes.")
        comm.Abort()
        sys.exit()

    with open(args.cfg_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        f.close()
    if rank == 0:
        main(
            config,
            args.seed,
            args.n_workers,
            args.generation_num,
            args.eval_ep_num,
            args.log,
            args.save_model_period,
        )
    else:
        worker(config["env"], config["network"])
