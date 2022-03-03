import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from main import *
plt.style.use(['science'])


def run_experiment(A, X, Nx, Na, num_sensors, dim) -> pd.DataFrame:
    """Get performance of differnet algorithms with given input data."""
    solver_performance_list = list()
    X_hat = SOCP(A, Nx, Na, num_sensors, dim)
    solver_performance_list.append({'solver': 'SOCP', 'mse': MSE(X, X_hat)})

    X_hat = SDP(A, Nx, Na, num_sensors, dim)
    solver_performance_list.append({'solver': 'SDP', 'mse': MSE(X, X_hat)})

    X_hat = noisy_SDP(A, Nx, Na, num_sensors, dim)
    solver_performance_list.append({'solver': 'noisy SDP', 'mse': MSE(X, X_hat)})

    X_hat = steepest_descent_with_projection(A, X, Nx, Na, num_sensors, dim, tol=1E-7)
    solver_performance_list.append({'solver': 'projection', 'mse': MSE(X, X_hat)})

    return pd.DataFrame(solver_performance_list)


if __name__ == "__main__":
    num_trials = 2
    num_anchors = 3
    num_sensors = 10
    dim = 2

    #
    lst = list()
    for _ in tqdm(range(num_trials)):
        A, X = simulate_anchor_and_sensor(num_anchors, num_sensors, dim)
        A = np.array([[-1, -1], [0, 1], [1 ,-1]])
        Nx, Na = calculate_distance(A, X, np.inf)
        lst.append(run_experiment(A, X, Nx, Na, num_sensors, dim))
    df = pd.concat(lst)

    plt.close()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x='solver', y='mse', data=df, ax=ax)
    ax.set_yscale('log')
    fig.savefig('./out/temp.png')

    #
    lst = list()
    N_size = list()
    for _ in tqdm(range(num_trials), desc='Random Data'):
        A, X = simulate_anchor_and_sensor(num_anchors, num_sensors, dim)
        A = np.array([[-1, -1], [0, 1], [1 ,-1]])
        for threshold in tqdm(np.linspace(0.1, 3, 5), leave=False, desc='Threshold'):
            Nx, Na = calculate_distance(A, X, threshold)
            N_size.append({'threshold': threshold, '|Nx|': len(Nx), '|Na|': len(Na)})
            if len(Nx) + len(Na) == 0:
                continue
            out = run_experiment(A, X, Nx, Na, num_sensors, dim)
            out['threshold'] = threshold
            lst.append(out)

    df_N_size = pd.DataFrame(N_size)
    df_N_size.to_csv('./out/N_size_by_threshold.csv', index=False)

    plt.close()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.lineplot(x='threshold', y='|Nx|', data=df_N_size, label='$|N_x|$', ax=ax)
    sns.lineplot(x='threshold', y='|Na|', data=df_N_size, label='$|N_a|$', ax=ax)
    ax.legend()
    ax.set_ylabel('Size of Constraints')
    fig.savefig('./out/N_size_by_threshold.png')

    df = pd.concat(lst)
    df['mse'] = df['mse'].astype(float)
    df['log_mse'] = np.log(df['mse'])
    df = df.reset_index(drop=True)
    df.to_csv('./out/mse_by_threshold.csv', index=False)

    plt.close()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.lineplot(x='threshold', y='log_mse', hue='solver', ci='sd', data=df, ax=ax)
    sns.despine()
    # ax.set_yscale('log')
    fig.savefig('./out/mse_by_threshold.png')
