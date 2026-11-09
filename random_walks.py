import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=22)

def random_walks(M, NTrials, dim):
    stds = []
    for m in M:
        walks = rng.random((NTrials, dim, m))*2 - 1
        walks /= np.linalg.norm(walks, axis=1, keepdims=True)
        finalPos = np.sum(walks, axis=2)
        stds.append(np.std(finalPos))
    return finalPos, stds

M = [100, 500, 1000, 5000, 10000]
N=10000
deviations = []
for m in M:
    finalPos, std = random_walks([m], N, 1)
    plt.hist(finalPos, bins=30, density=True)
    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title(f"Random Walks: {m} steps")
    plt.show()
    deviations.append(np.std(finalPos))


plt.plot(M, deviations, 'o-')
plt.plot(M, np.sqrt(M), 'r--', label='Theoretical std dev')
plt.xlabel("Number of steps")
plt.ylabel("Standard deviation")
plt.title("Random Walks: Standard deviation vs. number of steps")
plt.show()
