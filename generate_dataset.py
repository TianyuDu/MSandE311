"""
Script for generating data.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.linalg import norm
from tqdm import tqdm
plt.style.use('science')


def simulate_anchor_and_sensor(num_anchors: int, num_sensors: int, dim: int=2):
    MAP_SIZE = 1
    A = np.random.uniform(low=-MAP_SIZE, high=MAP_SIZE, size=(num_anchors, dim))
    X = np.random.uniform(low=-MAP_SIZE, high=MAP_SIZE, size=(num_sensors, dim))
    return A, X


def calculate_distance(A: np.ndarray, X: np.ndarray, threshold: float=np.inf):

    # construct known distances among sensors.
    Nx = dict()
    for i in range(X.shape[0]):
        xi = X[i, :]
        for j in range(X.shape[0]):
            xj = X[j, :]
            if (i < j) and (norm(xi - xj, 2) ** 2 < threshold):
                Nx[(i, j)] = norm(xi - xj, 2) ** 2

    # construct known anchor-sensor distances.
    Na = dict()
    for k in range(A.shape[0]):
        ak = A[k, :]
        for j in range(X.shape[0]):
            xj = X[j, :]
            if norm(ak - xj, 2) ** 2 < threshold:
                Na[(k, j)] = norm(ak - xj, 2) ** 2

    return Nx, Na


def generate_dataset(threshold=np.inf):
    _, X = simulate_anchor_and_sensor(num_anchors=3, num_sensors=10, dim=2)
    A = np.array([[-1, -1], [0, 1], [1 ,-1]])
    Nx, Na = calculate_distance(A, X, threshold)
    return A, X, Nx, Na


if __name__ == '__main__':
    # examine relationship between different threshold levels and |Nx| and |Na|.
    N_size = list()
    N = 100
    for threshold in tqdm(np.linspace(0, 9, num=100)):
        for _ in range(N):
            A, X, Nx, Na = generate_dataset(threshold)
            N_size.append({'threshold': threshold, '|Nx|': len(Nx), '|Na|': len(Na)})

    df_N_size = pd.DataFrame(N_size)
    df_N_size.to_csv('./out/N_size_by_threshold.csv', index=False)

    plt.close()
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.lineplot(x='threshold', y='|Nx|', data=df_N_size, label='$|N_x|$', ax=ax)
    sns.lineplot(x='threshold', y='|Na|', data=df_N_size, label='$|N_a|$', ax=ax)
    ax.legend()
    ax.set_ylabel('Size of Constraints')
    fig.savefig('./out/N_size_by_threshold.png', dpi=200)
