import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=22)

def random_walks(NTrials):
    M = [100, 500, 1000, 5000, 10000]
    stds = []
    for m in M:
        walks = rng.random((NTrials, m))*2 - 1
        finalPos = np.sum(walks, axis=1)
        plt.hist(finalPos, bins=30, density=True)
        plt.xlabel("x")
        plt.ylabel("Probability density")
        plt.title(f"Random Walks: {m} steps")
        plt.show()
        stds.append(np.std(finalPos))

    plt.plot(M, stds, 'o-')
    plt.plot(M, np.sqrt(M), 'r--', label='Theoretical std dev')
    plt.xlabel("Number of steps")
    plt.ylabel("Standard deviation")
    plt.title("Random Walks: Standard deviation vs. number of steps")
    plt.show()

random_walks(10000)


