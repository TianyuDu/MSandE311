import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import algorithms
import generate_dataset

plt.style.use(['science'])


def MSE(X_true, X_pred):
    return np.mean((X_true - X_pred) ** 2)


def run_experiment(A, X, Nx, Na, num_sensors, dim) -> pd.DataFrame:
    """Get performance of differnet algorithms with given input data."""
    solver_performance_list = list()

    X_hat = algorithms.SOCP(A, Nx, Na, num_sensors, dim)
    solver_performance_list.append({'solver': 'SOCP', 'MSE': MSE(X, X_hat)})

    X_hat = algorithms.SDP(A, Nx, Na, num_sensors, dim)
    solver_performance_list.append({'solver': 'SDP', 'MSE': MSE(X, X_hat)})

    X_hat = algorithms.least_square(A, Nx, Na, num_sensors, dim, verbose=False)
    solver_performance_list.append({'solver': 'LS', 'MSE': MSE(X, X_hat)})

    return pd.DataFrame(solver_performance_list)


if __name__ == "__main__":
    # configure the simulations.
    num_anchors = 3
    num_sensors = 10
    dim = 2

    num_trials = 10
    threshold_range = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 8.0]

    # comparsing performance
    performance_list = list()
    for _ in tqdm(range(num_trials), desc='Random Data'):
        for threshold in tqdm(threshold_range, leave=False, desc='Threshold'):
            A, X, Nx, Na = generate_dataset.generate_dataset(threshold)
            if len(Nx) == 0 or len(Na) == 0:
                continue
            p = run_experiment(A, X, Nx, Na, num_sensors, dim)
            p['threshold'] = threshold
            performance_list.append(p)

    df = pd.concat(performance_list).reset_index(drop=True)
    df['MSE'] = df['MSE'].astype(float)
    df['LOG_MSE'] = np.log(df['MSE'])

    df.to_csv('./out/Q1_performance.csv', index=False)

    plt.close()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.lineplot(x='threshold', y='LOG_MSE', hue='solver', ci='sd', data=df, ax=ax)
    fig.savefig('./out/log_mse_by_threshold.png')

    mean = df.groupby(['threshold', 'solver'])['LOG_MSE'].mean().reset_index().pivot('threshold', 'solver')
    std = df.groupby(['threshold', 'solver'])['LOG_MSE'].std().reset_index().pivot('threshold', 'solver')

    table = mean.round(3).astype(str) + '<pm>' + std.round(2).astype(str)
    print(table.to_latex().replace('<pm>', '$\\pm$'))
