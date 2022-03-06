import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import algorithms
import generate_dataset


def MSE(X_true, X_pred):
    return np.mean((X_true - X_pred) ** 2)


def generate_noisy_dataset(threshold, eps):
    A, X, Nx, Na = generate_dataset.generate_dataset(threshold)

    for key, val in Nx.items():
        multiplier = np.random.uniform(1-eps, 1+eps)
        # Nx[key] += (np.sqrt(Nx[key]) + np.random.randn() * 0.001) ** 2
        Nx[key] *= multiplier

    for key, val in Na.items():
        # Na[key] += (np.sqrt(Na[key]) + np.random.randn() * 0.001) ** 2
        multiplier = np.random.uniform(1-eps, 1+eps)
        Na[key] *= multiplier

    return A, X, Nx, Na


if __name__ == '__main__':
    num_anchors = 3
    num_sensors = 10
    dim = 2

    num_trials = 10
    EPS = 0.01
    threshold_range = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 8.0]

    # ==================================================================================================================
    #
    # ==================================================================================================================
    # performance_list = list()
    # for _ in tqdm(range(num_trials), desc='Random Data'):
    #     for threshold in tqdm(threshold_range, leave=False, desc='Threshold'):
    #         A, X, Nx, Na = generate_noisy_dataset(threshold, EPS)
    #         if len(Nx) == 0 or len(Na) == 0:
    #             continue

    #         # method 1: noisy SDP solution initialization.
    #         X_init = algorithms.noisy_SDP(A, Nx, Na, num_sensors, dim)
    #         X_hat = algorithms.least_square(A, Nx, Na, num_sensors, dim, verbose=False, X_init=X_init)
    #         performance_list.append({'threshold': threshold, 'solver': 'LS with noisy SDP initialization', 'MSE': MSE(X, X_hat)})

    #         # method 2: SOCP initialization.
    #         X_init = algorithms.SOCP(A, Nx, Na, num_sensors, dim)
    #         X_hat = algorithms.least_square(A, Nx, Na, num_sensors, dim, verbose=False, X_init=X_init)
    #         performance_list.append({'threshold': threshold, 'solver': 'LS with SOCP initialization', 'MSE': MSE(X, X_hat)})

    #         # method 3: Gaussian initialization.
    #         X_hat = algorithms.least_square(A, Nx, Na, num_sensors, dim, verbose=False, X_init=None)
    #         performance_list.append({'threshold': threshold, 'solver': 'LS with Gaussian initialization', 'MSE': MSE(X, X_hat)})


    # df = pd.DataFrame(performance_list).reset_index(drop=True)

    # # visualize.
    # df['MSE'] = df['MSE'].astype(float)
    # df['LOG_MSE'] = np.log(df['MSE'])
    # df.to_csv('./out/Q3.csv', index=False)

    # plt.close()
    # fig, ax = plt.subplots(figsize=(6, 4))
    # sns.lineplot(x='threshold', y='LOG_MSE', hue='solver', ci='sd', data=df, ax=ax)
    # fig.savefig('./out/Q3_comparison.png', dpi=300)

    # mean = df.groupby(['threshold', 'solver'])['LOG_MSE'].mean().reset_index().pivot('solver', 'threshold').round(3).astype(str)
    # std = df.groupby(['threshold', 'solver'])['LOG_MSE'].std().reset_index().pivot('solver', 'threshold').round(2).astype(str)
    # table = mean + '<pm>' + std
    # print(table.to_latex().replace('<pm>', '$\\pm$'))

    # ==================================================================================================================
    # with different levels of epsilon, heatmap of (epsilon, threshold) with diff-performance levels.
    # ==================================================================================================================
    performance_list = list()
    for eps in [0, 0.005, 0.01, 0.03, 0.05, 0.1, 0.15, 0.3]:
        for _ in tqdm(range(3), desc='Random Data'):
            for threshold in tqdm(threshold_range, leave=False, desc='Threshold'):
                A, X, Nx, Na = generate_noisy_dataset(threshold, eps=eps)
                if len(Nx) == 0 or len(Na) == 0:
                    continue

                # method 1: noisy SDP solution initialization.
                X_init = algorithms.noisy_SDP(A, Nx, Na, num_sensors, dim)
                X_hat = algorithms.least_square(A, Nx, Na, num_sensors, dim, verbose=False, X_init=X_init)
                performance_list.append({'threshold': threshold,
                                         'epsilon': eps,
                                         'solver': 'LS with noisy SDP initialization',
                                         'MSE': MSE(X, X_hat)})

                # method 2: Gaussian initialization.
                X_hat = algorithms.least_square(A, Nx, Na, num_sensors, dim, verbose=False, X_init=None)
                performance_list.append({'threshold': threshold,
                                         'epsilon': eps,
                                         'solver': 'LS with Gaussian initialization',
                                         'MSE': MSE(X, X_hat)})

    df_tau_eps = pd.DataFrame(performance_list).reset_index(drop=True)
    df_tau_eps['MSE'] = df_tau_eps['MSE'].astype(float)
    df_tau_eps['LOG_MSE'] = np.log(df_tau_eps['MSE'])
    df_tau_eps.to_csv('./out/Q3_tau_eps.csv', index=False)

    breakpoint()

    p_nsdp = df_tau_eps[df_tau_eps['solver'] == 'LS with noisy SDP initialization'].groupby(['epsilon', 'threshold'])['LOG_MSE'].mean()
    p_gaussian = df_tau_eps[df_tau_eps['solver'] == 'LS with Gaussian initialization'].groupby(['epsilon', 'threshold'])['LOG_MSE'].mean()

    diff = p_nsdp - p_gaussian
    diff = diff.reset_index().pivot('epsilon', 'threshold')


    p_nsdp = p_nsdp.reset_index().pivot('epsilon', 'threshold')
    p_gaussian = p_gaussian.reset_index().pivot('epsilon', 'threshold')
    # fig.savefig('./out/Q3_tau_eps_diff.png', dpi=300)

    # fig, ax = plt.subplots(figsize=(6, 6))
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
    # fig.savefig('./out/Q3_tau_eps_nsdp.png', dpi=300)

    # fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(p_nsdp, square=True, ax=axes[0])
    axes[0].set_title('log-MSE of noisy SDP initialization')
    sns.heatmap(p_gaussian, square=True, ax=axes[1])
    axes[1].set_title('log-MSE of Gaussian initialization')
    sns.heatmap(diff, square=True, ax=axes[2])
    axes[2].set_title('log-MSE(noisy SDP) - log-MSE(Gaussian)')
    fig.savefig('./out/Q3_tau_eps_all.png', dpi=300)
