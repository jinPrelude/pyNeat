import numpy as np


def crossover_offsprings(parents, rewards, offspring_num):
    offsprings = []
    p = np.arange(1, len(parents) + 1)[::-1] / sum(range(len(parents) + 1))
    for _ in range(offspring_num):
        p1_idx, p2_idx = np.random.choice(range(len(parents)), 2, p=p, replace=False)
        p1, p2 = parents[p1_idx], parents[p2_idx]
        if rewards[p1_idx] < rewards[p2_idx]:
            child = p1.crossover(p2)
        elif rewards[p1_idx] > rewards[p2_idx]:
            child = p2.crossover(p1)
        else:
            child = p1.crossover(p2, draw=True)
        offsprings.append(child)

    return offsprings


def mutate_offsprings(offsprings):
    for off in offsprings:
        off.mutate()
    return offsprings
