
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
        plt.savefig(f"{dim}DRandomWalk_{labels[i]}_{M}.png", dpi=300, bbox_inches="tight")
        plt.show()
    if dim > 1:
        r = np.linalg.norm(finalPos, axis=1)
        plt.figure()
        plt.hist(r, bins=30, density=True)
        plt.xlabel("r")
        plt.ylabel("Probability density")
        plt.title(f"{dim}D Random Walk: {M} steps")
        plt.savefig(f"{dim}DRandomWalk_r_{M}.png", dpi=300, bbox_inches="tight")
        plt.show()
    if dim == 2:
        plt.figure()
        plt.hist2d(finalPos[:, 0], finalPos[:, 1], bins=30, density=True)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"2D Random Walk: {M} steps")
        plt.colorbar(label="Probability density")
        plt.savefig(f"2DRandomWalk_xy_{M}.png", dpi=300, bbox_inches="tight")
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
plt.savefig("1DRandomWalk.png", dpi=300, bbox_inches="tight")
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

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

for dim in [1, 2, 3]:
    deviations = []
    distances = []

    for m in M:
        finalPos, std = random_walks(m, N, dim)
        deviations.append(np.std(finalPos[:, 0]))
        r = np.linalg.norm(finalPos, axis=1)
        distances.append(np.mean(r))

    ax1.plot(M, deviations, 'o-', label=f'{dim}D')
    ax2.plot(M, distances, 'o-', label=f'{dim}D')

ax1.set_xlabel("Number of steps")
ax1.set_ylabel("Standard deviation of x")
ax1.set_title("Coordinate Standard Deviation vs. Steps")
ax1.legend()
fig1.savefig("CoordinateStdDev.png", dpi=300, bbox_inches="tight")

ax2.set_xlabel("Number of steps")
ax2.set_ylabel("Mean distance")
ax2.set_title("Mean Distance vs. Steps")
ax2.legend()
fig2.savefig("MeanDistance.png", dpi=300, bbox_inches="tight")

plt.show()
