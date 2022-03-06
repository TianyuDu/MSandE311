import cvxpy as cp
import numpy as np
import torch
from torch import nn
from torch.optim import SGD


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


def least_square(A: np.ndarray, Nx: dict, Na: dict, num_sensors: int, dim: int=2, verbose=False, X_init=None):
    A = torch.Tensor(A)
    if X_init is None:
        X = nn.Parameter(torch.randn((num_sensors, dim)), requires_grad=True)
    else:
        X = nn.Parameter(torch.Tensor(X_init), requires_grad=True)

    optimizer = SGD([X], lr=0.001)

    MAX_EPOCHS = 5000
    for epoch in range(MAX_EPOCHS):

        optimizer.zero_grad()
        loss = torch.scalar_tensor(0.0)
        for (i, j), d2 in Nx.items():
            xi, xj = X[i, :], X[j, :]
            loss += (torch.norm((xi - xj), p=2)**2 - d2)**2

        for (k, j), d2 in Na.items():
            ak, xj = A[k, :], X[j, :]
            loss += (torch.norm((ak - xj), p=2)**2 - d2)**2

        loss.backward()
        optimizer.step()
        if verbose:
            print('Epoch=', epoch, 'Loss=', loss)

    return X.detach().numpy()


def steepest_descent_with_projection(A: np.ndarray, X: np.ndarray, Nx: dict, Na: dict, num_sensors: int, dim: int=2, verbose: bool=False, tol: float=1E-7):
    d = dim
    n = num_sensors
    beta = 30
    # Z = cp.Variable((d+n, d+n), PSD=True)  # PSD implies symmetric.
    # initialize Z.
    Z_init = np.random.randn(d+n, d+n)
    Z_init = (Z_init + Z_init.T).T  # for symmetric
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
    max_epochs = 100000

    while delta > tol:
    # for _ in range(100):
        if epoch > max_epochs:
            break
        X_before = Zk[-n:, :d]
        error_before = np.mean((X - X_before) ** 2)

        Zk_hat = Zk - 1/beta * grad_f(Zk)
        Lambda, V = np.linalg.eig(Zk_hat)

        # sort eigenvalues from large to small.
        idx = Lambda.argsort()[::-1]
        Lambda = Lambda[idx]
        V = V[:,idx]

        # only keep the largest eigenvalues.
        top = 6
        Lambda[top:] = 0

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
