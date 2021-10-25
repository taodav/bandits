import numpy as np
import matplotlib.pyplot as plt


def plot_regret(regret: np.ndarray):
    x = np.arange(regret.shape[0])
    fig, ax = plt.figure(), plt.axes()

    ax.plot(x, regret)

    ax.set_xlabel('Episodes')
    ax.set_ylabel(f"Regret", rotation=0, labelpad=65)
    # ax.set_title(f'Discounted episodic returns over environment steps (Prefilled ratio = 0.0)')
    plt.legend(bbox_to_anchor=(1.00, 1), loc='upper left')
