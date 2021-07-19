from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def loop():
    data = {"a": 7, "b": 3.14}
    comm.send(data, dest=1, tag=11)
    print(f"send data {data}")


def worker():
    data = comm.recv(source=0, tag=11)
    print(f"received data {data}")


if rank == 0:
    loop()
elif rank == 1:
    worker()
