import pickle as pkl

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def load_data(data_file_path):
    # ASSUME: data is a dictionary with two entries, "authors" and "papers"
    # "authors": list of dictionaries each representing an author's data. Each dictionary must have a field "embedding",
    #           which is a list of the embeddings of the author's papers.
    # "papers": list of paper embeddings
    with open(data_file_path, 'rb') as file:
        data = pkl.load(file)

    authors = data['authors']
    papers = data['papers']

    return authors, papers

def compute_cosine_similarity(papers, author_papers):
    return cosine_similarity(author_papers, papers)

def condense_author_scores(author_similarities):
    return np.max(author_similarities, axis=0)

def compute_scores(authors, paper_embeddings):
    paper_counts = []
    scores = []
    for author_data in authors:
        # get the author's papers
        author_embedding = author_data['embedding']
        num_papers = author_embedding.shape[0]

        # compute the cosine similarity of each of the author's papers with each paper in the dataset
        author_similarities_by_paper = compute_cosine_similarity(paper_embeddings, author_embedding)

        # for each author-paper pair, let the similarity score be the max similarity between that paper and one of the author's papers
        author_scores = condense_author_scores(author_similarities_by_paper)

        scores.append(author_scores)
        paper_counts.append(num_papers)

    scores_array = np.vstack(scores)
    return paper_counts, scores_array

def sample_utility(m, n, sample_m,sample_n,W):
    rng = np.random.default_rng()

    users = rng.choice(m, size=sample_m, replace=False)
    items = rng.choice(n, size=sample_n, replace=False)
    return W[users][:,items]

def sample_setting(M, N, m,n, W):
    rng = np.random.default_rng()
    sampled_authors_idxs = rng.choice(M, size=m, replace=False)
    sampled_papers_idxs = rng.choice(N, size=n, replace=False)
    return  W[sampled_authors_idxs][:,sampled_papers_idxs]