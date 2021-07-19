from mpi4py import MPI
import os
import subprocess
import sys
import argparse
import random
import time

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


def loop(n_workers: int):
    for i in range(3):
        # send to workers
        for id in range(1, n_workers + 1):
            data = {"a": id, "b": 3.14}
            comm.send(data, dest=id, tag=1000)

        # receive from workers
        for _ in range(id):
            req = comm.irecv(tag=1001)
            data = req.wait()
            print(f"{data}")

    for id in range(1, n_workers + 1):
        data = "terminate"
        comm.send(data, dest=id, tag=1000)


def worker():
    count = 0
    while True:
        data = comm.recv(source=0, tag=1000)
        if data == "terminate":
            print(f"rank {rank} count: {count}")
            break
        count += 1
        print(f"data: {data} count: {count}")
        result = data["a"] * data["b"]
        comm.isend(result, dest=0, tag=1001)


def main(args):
    if rank == 0:
        loop(args.n_workers)
    else:
        worker()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", type=int, default=3)
    args = parser.parse_args()
    if "parent" == mpi_fork(args.n_workers + 1):
        sys.exit()
    main(args)
