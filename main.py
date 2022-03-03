from typing import Optional

import torch
import torch.nn as nn
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.linalg import norm

# TODO: move solvers to a separate script.
def MSE(X_true, X_pred):
    return np.mean((X_true - X_pred) ** 2)


def SOCP(A: np.ndarray, Nx: dict, Na: dict, num_sensors: int, dim: int=2):
    # unknown locations.
    X_var = cp.Variable((num_sensors, dim))
    objective = cp.Minimize(0)
    constraints = list()
    for (i, j), d in Nx.items():
        constraints.append(
            cp.atoms.norm2(X_var[i, :] - X_var[j, :]) <= np.sqrt(d)
        )

    for (k, j), d in Na.items():
        constraints.append(
            cp.atoms.norm2(A[k, :] - X_var[j, :]) <= np.sqrt(d)
        )

    prob = cp.Problem(objective, constraints)

    result = prob.solve(verbose=False)
    return X_var.value


def SDP(A: np.ndarray, Nx: dict, Na: dict, num_sensors: int, dim: int=2):
    d = dim
    n = num_sensors
    Z = cp.Variable((d+n, d+n), PSD=True)  # PSD implies symmetric.
    objective = cp.Minimize(0)
    constraints = [Z[:d, :d] == np.eye(d)]

    for (i, j), distance_squared in Nx.items():
        ei = np.zeros((n, 1))
        ei[i] = 1

        ej = np.zeros((n, 1))
        ej[j] = 1

        vec = np.concatenate([np.zeros((d, 1)), ei - ej], axis=0)
        constraints.append(
            cp.atoms.sum(cp.atoms.multiply(vec @ vec.T, Z)) == distance_squared
        )

    for (k, j), distance_squared in Na.items():
        ak = A[k, :].reshape(-1, 1)
        ej = np.zeros((n, 1))
        ej[j] = 1
        vec = np.concatenate([ak, -ej], axis=0)
        constraints.append(
            cp.atoms.sum(cp.atoms.multiply(vec @ vec.T, Z)) == distance_squared
        )

    prob = cp.Problem(objective, constraints)
    result = prob.solve(verbose=False)

    X_hat = Z.value[-n:, :d]
    return X_hat


# def LS(A: np.ndarray, Nx: dict, Na: dict, num_sensors: int, dim: int=2):
#     # TODO: Fix this.
#     X_var = cp.Variable((num_sensors, dim))
#     objective_func = 0
#     for (i, j), d in Nx.items():
#         objective_func += cp.atoms.square(
#             cp.atoms.square(cp.atoms.norm2(X_var[i, :] - X_var[j, :])) - d
#         )

#     for (k, j), d in Na.items():
#         objective_func += cp.atoms.square(
#             cp.atoms.square(cp.atoms.norm2(A[k, :] - X_var[j, :])) - d
#         )

#     objective = cp.Minimize(objective_func)
#     prob = cp.Problem(objective)
#     result = prob.solve()


# def LS_torch(A: np.ndarray, Nx: dict, Na: dict, num_sensors: int, dim: int=2):
#     A = torch.Tensor(A)
#     X_initial = torch.Tensor(torch.zeros(num_sensors, dim))
#     X_var = nn.Parameter(X_initial)
#     objective_func = torch.scalar_tensor(0.0)
#     for (i, j), d in Nx.items():
#         objective_func += ((torch.norm(X_var[i, :] - X_var[j, :], p=2)**2) - d)**2

#     for (k, j), d in Na.items():
#         objective_func += ((torch.norm(A[k, :] - X_var[j, :], p=2)**2) - d)**2

#     torch.optim.SGD()

#     objective = cp.Minimize(objective_func)
#     prob = cp.Problem(objective)
#     result = prob.solve(requires_grad=True)
#     # TODO: complete this.


def steepest_descent_with_projection(A: np.ndarray, X: np.ndarray, Nx: dict, Na: dict, num_sensors: int, dim: int=2, verbose: bool=False, tol: float=1E-7):
    d = dim
    n = num_sensors
    beta = 30
    # Z = cp.Variable((d+n, d+n), PSD=True)  # PSD implies symmetric.
    # initialize Z.
    Z_init = np.zeros((d+n, d+n))
    # objective = cp.Minimize(0)
    # constraints = [Z[:d, :d] == np.eye(d)]
    Z_init[:d, :d] = np.eye(d)

    # construct the m Ai's.
    A_list = list()
    b_list = list()

    # add constraint Z[:d, :d] == np.eye(d)
    e = np.zeros((n+d, 1))
    e[0] = 1
    A_list.append(e @ e.T)
    b_list.append(1)

    e = np.zeros((n+d, 1))
    e[1] = 1
    A_list.append(e @ e.T)
    b_list.append(1)


    e = np.zeros((n+d, 1))
    e[0] = 1
    e[1] = 1
    A_list.append(e @ e.T)
    b_list.append(2)


    for (i, j), distance_squared in Nx.items():
        ei = np.zeros((n, 1))
        ei[i] = 1

        ej = np.zeros((n, 1))
        ej[j] = 1

        vec = np.concatenate([np.zeros((d, 1)), ei - ej], axis=0)
        A_list.append(vec @ vec.T)
        b_list.append(distance_squared)

    for (k, j), distance_squared in Na.items():
        ak = A[k, :].reshape(-1, 1)
        ej = np.zeros((n, 1))
        ej[j] = 1
        vec = np.concatenate([ak, -ej], axis=0)
        A_list.append(vec @ vec.T)
        b_list.append(distance_squared)

    b = np.array(b_list)

    def grad_f(Zk):
        grad = np.zeros_like(Zk)
        for i, Ai in enumerate(A_list):
            Ax_minus_b = (Ai * Zk).sum() - b[i]
            grad += Ai * Ax_minus_b
        return grad

    Zk = Z_init
    delta = 1E9
    epoch = 0

    while delta > tol:
        X_before = Zk[-n:, :d]
        error_before = np.mean((X - X_before) ** 2)

        Zk_hat = Zk - 1/beta * grad_f(Zk)
        Lambda, V = np.linalg.eig(Zk_hat)
        Zk = V @ np.diag(np.maximum(Lambda, 0)) @ V.T


        X_after = Zk[-n:, :d]
        error_after = np.mean((X - X_after) ** 2)

        delta = np.abs(error_after - error_before)
        epoch += 1
        if verbose:
            print(epoch, error_after)

    return Zk[-n:, :d]


def noisy_SDP(A: np.ndarray, Nx: dict, Na: dict, num_sensors: int, dim: int=2):
    d = dim
    n = num_sensors
    Z = cp.Variable((d+n, d+n), PSD=True)  # PSD implies symmetric.

    delta_1 = cp.Variable(len(Nx))
    delta_2 = cp.Variable(len(Nx))

    delta_hat_1 = cp.Variable(len(Na))
    delta_hat_2 = cp.Variable(len(Na))

    objective = cp.Minimize(cp.sum(delta_1 + delta_2) + cp.sum(delta_hat_1 + delta_hat_2))
    constraints = [Z[:d, :d] == np.eye(d),
                   delta_1 >= 0,
                   delta_2 >= 0,
                   delta_hat_1 >= 0,
                   delta_hat_2 >= 0]

    for c, ((i, j), distance_squared) in enumerate(Nx.items()):
        ei = np.zeros((n, 1))
        ei[i] = 1

        ej = np.zeros((n, 1))
        ej[j] = 1

        vec = np.concatenate([np.zeros((d, 1)), ei - ej], axis=0)
        constraints.append(
            cp.atoms.sum(cp.atoms.multiply(vec @ vec.T, Z)) + delta_1[c] - delta_2[c] == distance_squared
        )

    for c, ((k, j), distance_squared) in enumerate(Na.items()):
        ak = A[k, :].reshape(-1, 1)
        ej = np.zeros((n, 1))
        ej[j] = 1
        vec = np.concatenate([ak, -ej], axis=0)
        constraints.append(
            cp.atoms.sum(cp.atoms.multiply(vec @ vec.T, Z)) + delta_hat_1[c] - delta_hat_2[c] == distance_squared
        )

    prob = cp.Problem(objective, constraints)
    result = prob.solve(verbose=False)

    X_hat = Z.value[-n:, :d]
    return X_hat



def plot(A: np.ndarray, X: np.ndarray, X_hat: Optional[np.ndarray]=None, title: Optional[str]=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(A[:, 0], A[:, 1], marker='o', label='anchors')
    ax.scatter(X[:, 0], X[:, 1], marker='x', label='sensors')
    if X_hat is not None:
        ax.scatter(X_hat[:, 0], X_hat[:, 1], marker='+', label='Sensor Fitted')
        # draw distance between actual sensor location and predicted sensor location.
        for i in range(X.shape[0]):
            x1, x2 = X[i, 0], X_hat[i, 0]
            y1, y2 = X[i, 1], X_hat[i, 1]
            ax.plot((x1, x2), (y1, y2), 'k-', color='pink', linewidth=0.5, alpha=1.0)
    ax.legend()
    if title is not None:
        ax.set_title(title)
    # TODO: update label and title.
    return fig, ax


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


if __name__ == '__main__':
    # simulate to generate true locations of sensors and anchors.
    num_anchors = 3
    num_sensors = 10
    dim = 2

    A, X = simulate_anchor_and_sensor(num_anchors, num_sensors, dim)
    A = np.array([[-1, -1], [0, 1], [1 ,-1]])
    print(f'{A.shape=:}, {X.shape=:}')
    fig, ax = plot(A, X)


    # TODO: experiment with different level of threshold.
    Nx, Na = calculate_distance(A, X, 1)
    # Nx, Na = calculate_distance(A, X, np.inf)
    print(f'{len(Nx)=:}, {len(Na)=:}')

    # optionally add noise.
    for key, val in Nx.items():
        multiplier = np.random.uniform(0.99, 1.01)
        # Nx[key] += (np.sqrt(Nx[key]) + np.random.randn() * 0.001) ** 2
        Nx[key] *= multiplier

    for key, val in Na.items():
        # Na[key] += (np.sqrt(Na[key]) + np.random.randn() * 0.001) ** 2
        multiplier = np.random.uniform(0.99, 1.01)
        Na[key] *= multiplier

    X_hat_SOCP = SOCP(A, Nx, Na, num_sensors, dim)
    fig, ax = plot(A, X, X_hat_SOCP)
    fig.savefig('./out/SOCP.png', bbox_inches='tight')
    print('SOCP', MSE(X, X_hat_SOCP))

    # LS(A, Nx, Na, num_sensors, dim)


    X_hat_SDP = SDP(A, Nx, Na, num_sensors, dim)
    fig, ax = plot(A, X, X_hat_SDP)
    fig.savefig('./out/SDP.png', bbox_inches='tight')
    print('SDP', MSE(X, X_hat_SDP))


    X_hat_noisy_SDP = noisy_SDP(A, Nx, Na, num_sensors, dim)
    fig, ax = plot(A, X, X_hat_noisy_SDP)
    fig.savefig('./out/noisy_SDP.png', bbox_inches='tight')
    print('noisy_SDP', MSE(X, X_hat_noisy_SDP))

    # Steepest Descent and Projection Method.
    X_hat_steepest_descent = steepest_descent_with_projection(A, X, Nx, Na, num_sensors, dim, verbose=False, tol=1E-7)
    print(MSE(X, X_hat_steepest_descent))

    solver_dict = {'SOCP': SOCP, 'SDP': SDP, 'noisy_SDP': noisy_SDP, 'steepest_descent': steepest_descent_with_projection}

    # TODO: refactor this.
    # compare performance of different algorithms.
    solver_performance_list = list()
    for solver_name, solver in solver_dict.items():
        X_hat = solver(A, Nx, num_sensors, dim)
        mse = np.mean((X - X_hat) ** 2)
        # solver_performance_list.append({'solver_name': solver_name, 'threshold': threshold, 'mse': mse})

        # for threshold in np.linspace(0.1, 3, 10):
            # Nx, Na = calculate_distance(A, X, threshold)
    performance = pd.DataFrame(solver_performance_list)
    sns.plot(x='threshold', y='mse', hue='solver_name', data=performance)
