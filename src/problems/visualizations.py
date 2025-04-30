from collections import defaultdict
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
    sns.set_palette(sns.color_palette("Paired", len(results.keys())))
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


def plot_groups_results_means_per_alpha(
    results: dict,
    groups_key: str,
    n_consumers: int,
    n_producers: int,
    n_runs: int,
    k_rec: int,
    save_path: Path,
) -> None:
    sns.set_style("whitegrid")

    alphas = results["all"].keys()

    print(f"Alphas: {alphas}")

    if not Path(save_path).exists():
        Path(save_path).mkdir(parents=True)

    plt.figure(figsize=(10, 6), dpi=300)
    sns.set_palette(sns.color_palette("Paired", len(results.keys())))

    results_per_gamma_p50 = defaultdict(lambda: defaultdict(list))
    results_per_gamma_p95 = defaultdict(lambda: defaultdict(list))
    results_per_gamma_p05 = defaultdict(lambda: defaultdict(list))

    for group, group_results in results.items():
        if group != "all":
            continue
        for alpha, alpha_results in group_results.items():
            for gamma, runs in alpha_results.items():
                for run in runs:
                    results_per_gamma_p50[gamma][alpha].append(np.percentile(run, axis=0, q=50))
                    results_per_gamma_p95[gamma][alpha].append(np.percentile(run, axis=0, q=95))
                    results_per_gamma_p05[gamma][alpha].append(np.percentile(run, axis=0, q=5))

    for gamma, alpha_results in results_per_gamma_p50.items():
        x_alphas = []
        p50s = []
        p95s = []
        p5s = []
        for alpha, runs in alpha_results.items():
            p50 = np.mean(runs, axis=0)
            p95 = np.mean(results_per_gamma_p95[gamma][alpha], axis=0)
            p5 = np.mean(results_per_gamma_p05[gamma][alpha], axis=0)
            x_alphas.append(alpha)
            p50s.append(p50)
            p95s.append(p95)
            p5s.append(p5)

        plt.plot(
            x_alphas,
            p50s,
            label=f"$\gamma$ {gamma}",
            marker="s",
            markersize=3,
        )
        plt.fill_between(
            x_alphas,
            p5s,
            p95s,
            alpha=0.3,
        )
        plt.xticks(x_alphas)
        plt.xlabel("$\\alpha$")
        plt.ylabel("Mean utility of population")

    plt.title(f'Mean utilities of population per alpha (for group "{groups_key}")')
    plt.legend(loc="upper right")
    plt.savefig(
        Path(save_path)
        / f"cvar_mean_{groups_key}_results_per_alpha_n_consumers_{n_consumers}_n_producers_{n_producers}_n_runs_{n_runs}_k_rec_{k_rec}.png",
    )


def plot_groups_results_std_per_alpha(
    results: dict,
    groups_key: str,
    n_consumers: int,
    n_producers: int,
    n_runs: int,
    k_rec: int,
    save_path: Path,
) -> None:
    sns.set_style("whitegrid")

    alphas = results["all"].keys()
    if not Path(save_path).exists():
        Path(save_path).mkdir(parents=True)

    plt.figure(figsize=(10, 6), dpi=300)
    sns.set_palette(sns.color_palette("colorblind", len(results.keys())))

    results_per_gamma = defaultdict(lambda: defaultdict(list))

    for group, group_results in results.items():
        if group == "all":
            continue
        for alpha, alpha_results in group_results.items():
            for gamma, runs in alpha_results.items():
                for i, run in enumerate(runs):
                    if len(results_per_gamma[gamma][alpha]) < i + 1:
                        results_per_gamma[gamma][alpha].append([np.mean(run)])
                    else:
                        results_per_gamma[gamma][alpha][i].append(np.mean(run))

    for gamma, alpha_results in results_per_gamma.items():
        x_alphas = []
        y_alphas = []
        errs = []
        for alpha, runs in alpha_results.items():
            run_stds = [np.std(run) for run in runs]
            mean_of_run_stds = np.mean(run_stds, axis=0)
            standard_error_of_mean = np.std(run_stds, axis=0) / np.sqrt(len(runs))
            x_alphas.append(alpha)
            y_alphas.append(mean_of_run_stds)
            errs.append(standard_error_of_mean)

        plt.plot(
            x_alphas,
            y_alphas,
            label=f"$\gamma$ {gamma}",
            marker="s",
            markersize=3,
        )
        plt.fill_between(
            x_alphas,
            np.array(y_alphas) - np.array(errs),
            np.array(y_alphas) + np.array(errs),
            alpha=0.2,
        )
        plt.xticks(x_alphas)
        plt.xlabel("$\\alpha$")
        plt.ylabel("Std. dev. of mean group utilities")

    plt.title(f'Std. dev. of mean group utilities per alpha (for group "{groups_key.replace("_", " ")}")')
    plt.legend(markerscale=0)
    plt.tight_layout()
    plt.savefig(
        Path(save_path)
        / f"cvar_{groups_key}_results_per_alpha_n_consumers_{n_consumers}_n_producers_{n_producers}_n_runs_{n_runs}_k_rec_{k_rec}.pdf",
    )
