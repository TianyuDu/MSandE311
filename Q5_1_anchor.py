import numpy as np
import pandas as pd
from tqdm import tqdm
import generate_dataset
import algorithms
import seaborn as sns
import matplotlib.pyplot as plt


def MSE(X_true, X_pred):
    return np.mean((X_true - X_pred) ** 2)


if __name__ == '__main__':
    # configure the simulations.
    num_sensors = 10
    dim = 2

    num_trials = 10
    threshold_range = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 8.0]

    performance_list = list()
    for _ in tqdm(range(num_trials), desc='Random Data'):
        for threshold in tqdm(threshold_range, leave=False, desc='Threshold'):
            A, X, Nx, Na = generate_dataset.generate_dataset_q5(threshold)
            if len(Nx) == 0 or len(Na) == 0:
                continue

            X_hat = algorithms.steepest_descent_with_projection(A, X, Nx, Na, num_sensors, dim, verbose=False)
            performance_list.append({'threshold': threshold, 'solver': 'Steepest Descent Projection', 'MSE': MSE(X, X_hat)})

    df = pd.DataFrame(performance_list)
    df['MSE'] = df['MSE'].astype(float)
    df['LOG_MSE'] = np.log(df['MSE'])
    df.to_csv('./out/Q5_1_anchor.csv', index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(x='threshold', y='LOG_MSE', ci='sd', data=df, ax=ax)
    fig.savefig('./out/Q5_1_anchor.png', dpi=300)

    mean = df.groupby('threshold')['LOG_MSE'].mean().round(3).astype(str)
    std = df.groupby('threshold')['LOG_MSE'].std().round(2).astype(str)

    table = mean + '<pm>' + std
    print(table.reset_index().T.to_latex().replace('<pm>', '$\\pm$'))
