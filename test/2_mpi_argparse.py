import argparse
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def loop(n_workers: int):
    for id in range(1, n_workers):
        data = {'a': id, 'b': 3.14}
        req = comm.isend(data, dest=id, tag=11)
        req.wait()
        print(f"send data {data}")

def worker():
    req = comm.irecv(source=0, tag=11)
    data = req.wait()
    print(f"received data {data}")

parser = argparse.ArgumentParser()
parser.add_argument("--n-workers", type=int, default=3)

if __name__=="__main__":
    args = parser.parse_args()
    if rank == 0:
        loop(args.n_workers)
    elif rank >= 1:
        worker()