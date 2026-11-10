
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

rng = np.random.default_rng(seed=22)

def A(N=10000, M=1000, boundary=20):
    positions = []
    for i in range(N):
        x = 0
        steps = rng.choice([-1, 1], size=M)
        for step in steps:
            if boundary is not None and x + step > boundary:
                step = -step
            x += step
        positions.append(x)
    return np.array(positions)

def B1(N=10000, M=1000, boundary=20):
    positions = []
    while len(positions) < N:
        x = 0
        absorbed = False
        steps = rng.choice([-1, 1], size=M)
        for step in steps:
            if x + step > boundary:
                absorbed = True
                break
            x += step
        if not absorbed:
            positions.append(x)
    return np.array(positions)


def B2(N=10000, M=1000, boundary=20):
    positions = []
    absorbed = 0
    for i in range(N):
        x = 0
        steps = rng.choice([-1, 1], size=M)
        for step in steps:
            if x + step > boundary:
                absorbed += 1
                break
            x += step
        positions.append(x)
    return np.array(positions), absorbed / N


def C(N=10000, M=1000, boundary=20):
    positions = []
    for i in range(N):
        x = 0
        steps = rng.choice([-1, 1], size=M)
        for step in steps:
            if x == boundary + 1 and step == -1:
                step = 1
            x += step
        positions.append(x)
    return np.array(positions)


def D(N=10000, M=1000, boundary=5, p=0.5):
    positions = []
    for i in range(N):
        x = 0
        steps = rng.choice([-1, 1], size=M)
        for step in steps:
            if (x == boundary and step == 1) or (x == boundary + 1 and step == -1):
                if rng.random() >= p:
                    step = -step
            x += step
        positions.append(x)
    return np.array(positions)


if __name__ == "__main__":
    N = 10000
    M = 1000
    boundaries = [5, 20, 50]

    # Case A
    plt.figure()
    for boundary in [None, 5, 20, 50]:
        positions = A(N, M, boundary)
        plt.hist(positions, bins=30, density=True, alpha=0.5,
                 label=f"Boundary = {boundary}")
    plt.xlabel("Final position")
    plt.ylabel("Probability density")
    plt.title("Case A: Reflecting Walls")
    plt.legend()
    plt.savefig("Plots/CaseA.png")
 
    # Case B1
    plt.figure()
    for boundary in boundaries:
        positions = B1(N, M, boundary)
        plt.hist(positions, bins=30, density=True, alpha=0.5,
                 label=f"Boundary = {boundary}")
    plt.xlabel("Final position")
    plt.ylabel("Probability density")
    plt.title("Case B1: Survivors Only")
    plt.legend()
    plt.savefig("Plots/CaseB1.png")
  
    # Case B2
    survivors = B1(N, M, 20)
    stopped, fraction = B2(N, M, 20)
    bins = np.arange(min(survivors.min(), stopped.min()) - 0.5, 21.5, 1)
    plt.figure()
    plt.hist(survivors, bins=bins, density=True, alpha=0.7,
             label="B1: Survivors only")
    plt.hist(stopped, bins=bins, density=True, alpha=0.7,
             label="B2: Stopped walkers counted")
    plt.xlabel("Final position")
    plt.ylabel("Probability")
    plt.title("Case B2: Absorbing Boundary at x = 20")
    plt.legend()
    plt.savefig("Plots/CaseB2.png")

    print(f"Fraction absorbed: {fraction:.4f}")

    # Case C
    plt.figure()
    for boundary in [None, 5, 20, 50]:
        positions = A(N, M, None) if boundary is None else C(N, M, boundary)
        plt.hist(positions, bins=30, density=True, alpha=0.5,
                 label=f"Boundary = {boundary}")
    plt.xlabel("Final position")
    plt.ylabel("Probability density")
    plt.title("Case C: No-Return Cliffs")
    plt.legend()
    plt.savefig("Plots/CaseC.png")
  

    # Case D
    plt.figure()
    for p in [0, 0.1, 0.5, 1]:
        positions = D(N, M, 5, p)
        plt.hist(positions, bins=30, density=True, alpha=0.5,
                 label=f"p = {p}")
    plt.xlabel("Final position")
    plt.ylabel("Probability density")
    plt.title("Case D: Tunneling Through Wall at x = 5")
    plt.legend()
    plt.savefig("Plots/CaseD.png")
    plt.show()
