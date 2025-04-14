import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import notebooks.movielens.twosf_solver as solver
from notebooks.movielens.sample_mallows import get_mallows_utilities
from tqdm import tqdm


def sample_curves(phis, m, n, k, delta, num_pts, num_samples):
    all_pairs = []
    for phi in tqdm(phis):
        sample_pairs = []
        for sample in range(num_samples):
            U = get_mallows_utilities(m, n, phi)
            pairs = solver.get_user_curve(U, k, delta, num_pts)
            sample_pairs.append(pairs)
        all_pairs.append(np.array(sample_pairs))

    return all_pairs


def plot_curves(phis, all_pairs, fig_file_name):
    viridis = mpl.colormaps["viridis"]

    for phi, sample_pairs in zip(phis, all_pairs):
        color = viridis(phi)
        sp_mean = np.mean(sample_pairs[:, :, 1], axis=0)
        sp_std = np.std(sample_pairs[:, :, 1], axis=0, ddof=1)

        plt.plot(
            sample_pairs[0, :, 0], sp_mean, color=color, label=(r"$\phi =$ %.2f" % phi)
        )
        plt.fill_between(
            sample_pairs[0, :, 0],
            sp_mean - 2 * sp_std / np.sqrt(num_samples),
            sp_mean + 2 * sp_std / np.sqrt(num_samples),
            color=color,
            alpha=0.3,
        )

    plt.ylabel("Minimum normalized user utility")
    plt.xlabel(r"Minimum normalized item utility guaranteed ($\gamma_I$)")
    plt.legend()
    plt.savefig(fig_file_name)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int)
    parser.add_argument("--m", type=int)

    parser.add_argument("--curves", type=int)
    parser.add_argument("--curve_pts", type=int, default=100)

    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--delta", type=float, default=1)

    parser.add_argument("--ci", type=int, default=1)

    parser.add_argument("--df", type=str, help="File where data is stored")
    parser.add_argument("--ff", type=str, help="File to store the plot")

    parser.add_argument("--components", type=int)
    parser.add_argument("--clusters", type=int)

    args = parser.parse_args()

    n = args.n
    m = args.m

    num_curves = args.curves
    num_pts = args.curve_pts

    k = args.k
    delta = args.delta

    fig_file = args.ff

    return n, m, num_curves, num_pts, k, delta, fig_file


if __name__ == "__main__":
    n, m, num_samples, num_pts, k, delta, fig_file = parse_args()

    phis = np.linspace(0.1, 0.9, 5)

    curves = sample_curves(phis, m, n, k, delta, num_pts, num_samples)
    plot_curves(phis, curves, fig_file)
