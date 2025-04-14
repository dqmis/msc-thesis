import numpy as np

from notebooks.movielens.experiment_utils import load_data, compute_scores, sample_utility
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import notebooks.movielens.twosf_solver as solver

from tqdm import tqdm
import argparse

import matplotlib.pyplot as plt

def cluster_data(author_data, n_components, n_clusters):
    author_embeddings = np.squeeze(np.array([np.mean(author['embedding'], axis=0) for author in author_data]))

    # reduce dimensions
    pca_model = PCA(n_components = n_components)
    Z = pca_model.fit_transform(author_embeddings)

    # cluster
    kmeans_model = KMeans(n_clusters=n_clusters)
    labels = kmeans_model.fit_predict(Z)

    return labels

def sample_homogeneous_utilities(W, labels, group, sample_n, n_clusters):
    group_authors = np.nonzero(labels == group)[0]
    W_restricted = W[group_authors,:]

    m,n = W_restricted.shape
    H = sample_utility(m, n, m,sample_n,W_restricted)

    return H

def get_curves_homogeneous(W, sample_n, num_curves, num_pts, k, delta, n_clusters):
    # cluster authors based on embeddings
    labels = cluster_data(authors, n_components, n_clusters)

    all_empirical_pairs = []
    for i in tqdm(range(num_curves)):
        # randomly select a cluster of authors
        group = np.random.default_rng().choice(n_clusters)

        invalid = True
        while invalid:
            # pick a random subset of papers of size sample_n
            H = sample_homogeneous_utilities(W, labels,group, sample_n, n_clusters)
            invalid = not np.all(np.sum(H, axis=0)) # assume: no papers provides zero utility to all authors

        # compute tradeoff curve
        pairs = solver.get_user_curve(H,k,delta,num_pts)
        all_empirical_pairs.append(pairs)

    return np.array(all_empirical_pairs)

def get_curves(W, M,N, sample_m, sample_n, num_curves, num_pts, k, delta):
    all_empirical_pairs = []
    for i in tqdm(range(num_curves)):
        # randomly select sample_m authors and sample_n papers
        U = sample_utility(M,N,sample_m,sample_n,W)

        # compute tradeoff curves
        pairs = solver.get_user_curve(U,k,delta,num_pts)
        all_empirical_pairs.append(pairs)

    return np.array(all_empirical_pairs)

def plot_curves(all_empirical_pairs, all_empirical_pairs_homog, num_curves, figname, ci=True):
    if ci:
        gammas = all_empirical_pairs[0,:,0]
        mean_tradeoff = np.mean(all_empirical_pairs[:,:,1], axis=0)
        std_tradeoff = np.std(all_empirical_pairs[:,:,1], axis=0, ddof=1)
        plt.plot(gammas, mean_tradeoff, color='blue', label="Random users")
        plt.fill_between(gammas, mean_tradeoff - 2*std_tradeoff / np.sqrt(num_curves), mean_tradeoff + 2*std_tradeoff / np.sqrt(num_curves), color='blue', alpha=0.2)

        gammas_homog = all_empirical_pairs_homog[0,:,0]
        mean_tradeoff_homog = np.mean(all_empirical_pairs_homog[:,:,1], axis=0)
        std_tradeoff_homog = np.std(all_empirical_pairs_homog[:,:,1], axis=0, ddof=1)

        plt.plot(gammas_homog, mean_tradeoff_homog, color='red', label="Homogeneous users")
        plt.fill_between(gammas_homog, mean_tradeoff_homog - (2*std_tradeoff_homog / np.sqrt(num_curves)), mean_tradeoff_homog + (2*std_tradeoff_homog / np.sqrt(num_curves)), color='red', alpha=0.2)
    else:
        for pairs in all_empirical_pairs:
            plt.plot(pairs[:,0], pairs[:,1], color='blue', linestyle='dotted')

        mean_tradeoff = np.mean(all_empirical_pairs, axis=0)
        plt.plot(mean_tradeoff[:,0], mean_tradeoff[:,1], color='blue', label="Random users")

        for pairs in all_empirical_pairs_homog:
            plt.plot(pairs[:,0], pairs[:,1], color='red', linestyle='dotted')

        mean_tradeoff_homog = np.mean(all_empirical_pairs_homog, axis=0)
        plt.plot(mean_tradeoff_homog[:,0], mean_tradeoff_homog[:,1], color='red', label="Homogeneous users")

    plt.ylabel('Normalized user utility')
    plt.xlabel(r'Fraction of best min normalized item utility guaranteed, $\gamma^I$')
    plt.legend()
    plt.ylim([0,1.1])
    plt.savefig(figname)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int)
    parser.add_argument('--m', type=int)

    parser.add_argument('--curves', type=int)
    parser.add_argument('--curve_pts', type=int, default=100)

    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--delta', type=float, default=1)

    parser.add_argument('--ci', type=int, default=1)

    parser.add_argument('--df', type=str, help='File where data is stored')
    parser.add_argument('--ff', type=str, help='File to store the plot')

    parser.add_argument('--components', type=int)
    parser.add_argument('--clusters', type=int)

    args = parser.parse_args()

    sample_n = args.n
    sample_m = args.m

    num_curves = args.curves
    num_pts = args.curve_pts

    k = args.k
    delta = args.delta
    data_file = args.df
    fig_file = args.ff

    ci = args.ci

    n_components = args.components
    n_clusters = args.clusters

    return sample_n, sample_m, num_curves, num_pts, k, delta, data_file, fig_file, ci, n_components, n_clusters

if __name__ == '__main__':
    # get arguments
    sample_n, sample_m, num_curves, num_pts, k, delta, data_file, fig_file, ci, n_components, n_clusters = parse_args()

    # load data
    authors, papers = load_data(data_file)
    M = len(authors)
    N = papers.shape[0]

    # construct similarity scores from author/paper embeddings
    paper_counts, W = compute_scores(authors, papers)

    # get tradeoff curves for homogeneous subpopulations
    all_empirical_pairs_homog = get_curves_homogeneous(W, sample_n, num_curves, num_pts, k, delta, n_clusters)

    # get tradeoff curves for randomly selected subpopulations
    all_empirical_pairs = get_curves(W, M,N, sample_m, sample_n, num_curves, num_pts, k, delta)

    # plot tradeoff curves
    plot_curves(all_empirical_pairs,all_empirical_pairs_homog, num_curves, fig_file, ci=ci)