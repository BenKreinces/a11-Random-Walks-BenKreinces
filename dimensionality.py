
import numpy as np
import matplotlib.pyplot as plt
from random_walks import random_walks

def plot_random_walks(finalPos, dim, M):
    labels = ['x', 'y', 'z']
    for i in range(dim):
        plt.figure()
        plt.hist(finalPos[:, i], bins=30, density=True)
        plt.xlabel(labels[i])
        plt.ylabel("Probability density")
        plt.title(f"{dim}D Random Walk: {M} steps")
        plt.show()
    if dim > 1:
        r = np.linalg.norm(finalPos, axis=1)
        plt.figure()
        plt.hist(r, bins=30, density=True)
        plt.xlabel("r")
        plt.ylabel("Probability density")
        plt.title(f"{dim}D Random Walk: {M} steps")
        plt.show()
    if dim == 2:
        plt.figure()
        plt.hist2d(finalPos[:, 0], finalPos[:, 1], bins=30, density=True)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"2D Random Walk: {M} steps")
        plt.colorbar(label="Probability density")
        plt.show()

# Part 1
M = [100, 500, 1000, 5000, 10000]
N = 10000
deviations = []
dim = 1
for m in M:
    finalPos, std = random_walks(m, N, dim)
    plot_random_walks(finalPos, dim, m)
    deviations.append(np.std(finalPos[:, 0]))
plt.figure()
plt.plot(M, deviations, 'o-', label='Simulated')
plt.plot(M, np.sqrt(M), 'r--', label='Theoretical')
plt.xlabel("Number of steps")
plt.ylabel("Standard deviation")
plt.title("1D Random Walk: Standard Deviation")
plt.legend()
plt.show()

# Part 2
M = 1000
N = 10000
for dim in [2, 3]:
    finalPos, std = random_walks(M, N, dim)
    plot_random_walks(finalPos, dim, M)

# Part 2: Dimensional comparison
M = [100, 500, 1000, 5000, 10000]
N = 10000

plt.figure()
for dim in [1, 2, 3]:
    deviations = []
    distances = []

    for m in M:
        finalPos, std = random_walks(m, N, dim)
        deviations.append(np.std(finalPos[:, 0]))
        r = np.linalg.norm(finalPos, axis=1)
        distances.append(np.mean(r))
    plt.figure(1)
    plt.plot(M, deviations, 'o-', label=f'{dim}D')
    plt.figure(2)
    plt.plot(M, distances, 'o-', label=f'{dim}D')
    

plt.xlabel("Number of steps")
plt.ylabel("Standard deviation of x")
plt.title("Coordinate Standard Deviation vs. Steps")
plt.legend()

plt.figure(1)
plt.xlabel("Number of steps")
plt.ylabel("Mean distance")
plt.title("Mean Distance vs. Steps")
plt.legend()

plt.show()