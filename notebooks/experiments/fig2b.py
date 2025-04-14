from notebooks.movielens.experiment_utils import load_data, compute_scores, sample_utility

import argparse

import numpy as np

import notebooks.movielens.twosf_solver as solver

from tqdm import tqdm
import matplotlib.pyplot as plt

def form_misestimated_values(m1, m2, W_sample):
    W1 = W_sample[:m1]
    W2 = W_sample[m1:]

    W1hat = W1
    W2hat = np.repeat(np.expand_dims(np.mean(W1, axis=0),0), repeats=m2, axis=0)

    What_sample = np.concatenate((W1hat, W2hat), axis=0)
    return What_sample

def get_curves(W, M, N, m1, m2, m, n, n_pts, num_samples, k, delta):
    all_pairs = []
    all_pairs_corr = []
    W_samples = []
    for i in tqdm(range(num_samples)):
        # randomly select a subset of m authors and n papers
        W_sample = sample_utility(M, N, m,n, W)
        W_samples.append(W_sample)

        # form an "estimated" utility matrix in which m2 of the authors are assigned the average value of the other authors
        What_sample = form_misestimated_values(m1, m2, W_sample)

        # compute the user-item fairness tradeoff curve in the context of mis-estimation
        pairs = solver.get_user_curve(What_sample,k, delta,n_pts,just_value=False)
        # compute the counterfactual user-item fairness tradeoff curve if preferences had been correctly estimated
        pairs_corr = solver.get_user_curve(W_sample,k, delta,n_pts,just_value=False)

        all_pairs.append(pairs)
        all_pairs_corr.append(pairs_corr)

    return W_samples, all_pairs, all_pairs_corr

def get_group_specific_utilities(W_samples, all_pairs, m1):
    all_gammas = []
    all_v1s = []
    all_v2s = []
    for pairs, W_sample in zip(all_pairs, W_samples):
        gammas = []
        v1s = []
        v2s = []
        for pair in pairs:
            gamma_item, v_user_problem = pair

            # get optimal recommendation policy
            X = v_user_problem.variables()[0].value

            # compute normalized user utilities of each author under this policy
            normalized_user_utilities = np.sum(np.multiply(X,W_sample), axis=1) / np.max(W_sample, axis=1)

            # compute minimum normalized author utility in each group
            v1 = np.min(normalized_user_utilities[:m1])
            v2 = np.min(normalized_user_utilities[m1:])
            v1s.append(v1)
            v2s.append(v2)
            gammas.append(gamma_item)

        all_gammas.append(gammas)
        all_v1s.append(v1s)
        all_v2s.append(v2s)
    return all_gammas[0], np.array(all_v1s), np.array(all_v2s)

def plot_curves(gammas, v1s, v2s, v1s_corr, v2s_corr, num_samples, fig_file_name):
    # compute means/stds across random subsamples of authors and papers
    v1s_mean = np.mean(v1s, axis=0)
    v1s_std = np.std(v1s, axis=0, ddof=1)

    v2s_mean = np.mean(v2s, axis=0)
    v2s_std = np.std(v2s, axis=0, ddof=1)

    v1s_corr_mean = np.mean(v1s_corr, axis=0)
    v1s_corr_std = np.std(v1s_corr, axis=0, ddof=1)

    v2s_corr_mean = np.mean(v2s_corr, axis=0)
    v2s_corr_std = np.std(v2s_corr, axis=0, ddof=1)

    # Utility of the minimum user in the correctly estimated population
    # plt.plot(gammas, v1s_mean, label=r'Min user in $C$ under mis-estimation', color='blue')
    # plt.fill_between(gammas, v1s_mean - 2*v1s_std / np.sqrt(num_samples), v1s_mean + 2*v1s_std / np.sqrt(num_samples), color='blue', alpha=0.2)

    # Utility of the minimum user in the mis-estimated population
    plt.plot(gammas, v2s_mean, label=r'Min user in $M$ under mis-estimation', color='orange')
    plt.fill_between(gammas, v2s_mean - 2*v2s_std / np.sqrt(num_samples), v2s_mean + 2*v2s_std / np.sqrt(num_samples), color='orange', alpha=0.2)

    # Utility of the minimum user in the correctly estimated population if all utilities had actually been propertly estimated
    # plt.plot(gammas, v1s_corr_mean, label='Min user in $C$, true estimates', color='blue', linestyle='dashed')
    # plt.fill_between(gammas, v1s_corr_mean - 2*v1s_corr_std / np.sqrt(num_samples), v1s_corr_mean + 2*v1s_corr_std / np.sqrt(num_samples), color='blue', alpha=0.2)

    # Utility of the minimum user in the mis-estimated population if all utilities had actually been propertly estimated
    plt.plot(gammas, v2s_corr_mean, label='Min user in $M$, true estimates', color='orange', linestyle='dashed')
    plt.fill_between(gammas, v2s_corr_mean - 2*v2s_corr_std / np.sqrt(num_samples), v2s_corr_mean + 2*v2s_corr_std / np.sqrt(num_samples), color='orange', alpha=0.2)

    plt.legend()
    plt.xlabel(r'Fraction of best min normalized item utility guaranteed, $\gamma^I$')
    plt.ylabel('Normalized user utility')
    plt.ylim([0,1.1])
    plt.savefig(fig_file_name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int)
    parser.add_argument('--m', type=int)

    parser.add_argument('--beta', type=float)

    parser.add_argument('--curves', type=int)
    parser.add_argument('--curve_pts', type=int, default=100)

    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--delta', type=float, default=1)

    parser.add_argument('--df', type=str, help='File where data is stored')
    parser.add_argument('--ff', type=str, help='File to store the plot')

    args = parser.parse_args()

    sample_n = args.n
    sample_m = args.m

    beta = args.beta

    num_curves = args.curves
    num_pts = args.curve_pts

    k = args.k
    delta = args.delta

    data_file = args.df
    fig_file = args.ff

    return sample_n, sample_m, beta, num_curves, num_pts, k, delta, data_file, fig_file

if __name__ == "__main__":
    sample_n, sample_m, beta, num_samples, num_pts, k, delta, data_file, fig_file = parse_args()

    # compute size of each sub-population
    m1 = int(beta*sample_m)
    m2 = sample_m - m1

    authors, papers = load_data(data_file)
    M = len(authors)
    N = papers.shape[0]

    _, W = compute_scores(authors, papers)

    W_samples, all_pairs, all_pairs_corr = get_curves(W, M, N, m1, m2, sample_m, sample_n, num_pts, num_samples, k, delta)

    # post-process curves
    gammas, v1s, v2s = get_group_specific_utilities(W_samples, all_pairs, m1)
    _, v1s_corr, v2s_corr = get_group_specific_utilities(W_samples, all_pairs_corr, m1)

    # plot curves
    plot_curves(gammas, v1s, v2s, v1s_corr, v2s_corr, num_samples, fig_file)