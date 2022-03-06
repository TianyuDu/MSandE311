import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import algorithms
import generate_dataset

def MSE(X_true, X_pred):
    return np.mean((X_true - X_pred) ** 2)

if __name__ == '__main__':
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

            # get initialization for
            X_init = algorithms.SOCP(A, Nx, Na, num_sensors, dim)
            X_hat = algorithms.least_square(A, Nx, Na, num_sensors, dim, verbose=False, X_init=X_init)
            performance_list.append({'threshold': threshold, 'solver': 'LS_with_SOCP_initialization', 'MSE': MSE(X, X_hat)})

    df = pd.DataFrame(performance_list).reset_index(drop=True)
    df['MSE'] = df['MSE'].astype(float)
    df['LOG_MSE'] = np.log(df['MSE'])

    # save the result.
    df.to_csv('./out/Q2_performance.csv', index=False)

    # compare the performance against normal LS.
    df_q1 = pd.read_csv('./out/Q1_performance.csv')
    df = pd.concat([df, df_q1], axis=0)
    df = df.reset_index(drop=True)

    df = df[df['solver'].isin(['LS', 'LS_with_SOCP_initialization'])]
    # plot the result.
    mean = df.groupby(['threshold', 'solver'])['LOG_MSE'].mean().round(3).astype(str)
    std = df.groupby(['threshold', 'solver'])['LOG_MSE'].std().round(2).astype(str)

    table = mean + '<pm>' + std
    table = pd.pivot(table.reset_index(), 'solver', 'threshold')
    print(table.to_latex().replace('<pm>', '$\\pm$'))

    plt.close()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(x='threshold', y='LOG_MSE', hue='solver', ci='sd', data=df, ax=ax)
    fig.savefig('./out/Q2_comparison.png', dpi=300)
