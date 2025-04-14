from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_groups_results_per_gamma(
    results: dict,
    groups_key: str,
    n_consumers: int,
    n_producers: int,
    n_runs: int,
    k_rec: int,
    method: str,
    save_path: Path,
) -> None:
    sns.set_style("whitegrid")

    if not Path(save_path).exists():
        Path(save_path).mkdir(parents=True)

    gammas: set[float] = set()

    plt.figure(figsize=(10, 6), dpi=300)
    sns.set_palette(sns.color_palette("viridis", len(results.keys())))
    max_zorder = len(results.keys())
    zorder = 1
    for group_name, group_results in results.items():
        for alpha, alpha_results in group_results.items():
            gamma_means = {}
            gamma_std_err = {}
            for gamma, runs in alpha_results.items():
                gammas.add(gamma)
                run_means = [np.mean(run) for run in runs]
                gamma_means[gamma] = np.mean(run_means)
                gamma_std_err[gamma] = np.std(run_means) / np.sqrt(len(runs))

            if group_name == "all":
                plt.plot(
                    list(gamma_means.keys()),
                    list(gamma_means.values()),
                    color="black",
                    label="Mean",
                    linestyle="--",
                    marker="s",
                    zorder=max_zorder,
                    markersize=3,
                )
                continue

            plt.plot(
                list(gamma_means.keys()),
                list(gamma_means.values()),
                label=f"{group_name.replace('_', ' ').capitalize()}",
                marker="s",
                markersize=3,
                zorder=zorder,
            )
            plt.fill_between(
                list(gamma_std_err.keys()),
                np.array(list(gamma_means.values())) - np.array(list(gamma_std_err.values())),
                np.array(list(gamma_means.values())) + np.array(list(gamma_std_err.values())),
                alpha=0.1,
                zorder=zorder,
            )
            zorder += 1
            break

    plt.ylabel("Normalized consumer utility")
    plt.title(
        f'Consumer and producer utility tradeoff for retrieving k={k_rec} items (for group "{groups_key.replace("_", " ")}" )'
    )
    plt.xlabel(r"Fraction of best min producer utility guaranteed, $\gamma$")
    plt.legend(markerscale=0)
    plt.savefig(
        f"{save_path}/{method}_tradeoff_curve_group_{groups_key}_n_consumers_{n_consumers}_n_producers_{n_producers}_n_runs_{n_runs}_k_rec_{k_rec}_alpha_{alpha}.png"
    )
