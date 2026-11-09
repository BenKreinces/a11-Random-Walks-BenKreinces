import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=22)

def random_walks(m, NTrials, dim):
    stds = []
    walks = rng.random((NTrials, dim, m))*2 - 1
    walks /= np.linalg.norm(walks, axis=1, keepdims=True)
    finalPos = np.sum(walks, axis=2)
    stds.append(np.std(finalPos, axis = 0))
    return finalPos, stds
