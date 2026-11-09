import numpy as np
import matplotlib.pyplot as plt

M = 1000

rng = np.random.default_rng(seed=22)
vector = (rng.random((10, 2)) * 2 - 1) 
norm =vector / np.linalg.norm(vector, axis=1)[:, np.newaxis]

print(vector)