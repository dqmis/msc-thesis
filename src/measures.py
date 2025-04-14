from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def bce_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def plot_loss_per_group(test_df, group_col):
    loss_per_group = defaultdict(list)
    for group, group_df in test_df.groupby(group_col):
        loss = bce_loss(group_df["label"], group_df["probs"])
        loss_per_group[group] = loss

    plt.bar(loss_per_group.keys(), loss_per_group.values())
    plt.xlabel(" ".join(group_col.split("_")).capitalize())
    plt.xticks(rotation=90)
    plt.ylabel("BCE Loss")
    plt.title(f"BCE Loss per {group_col} (lower is better)")
    plt.show()


def user_utility_at_k(df: pd.DataFrame, k: int) -> pd.DataFrame:
    k = min(k, df.shape[0])
    ratings = df.rating.sort_values(ascending=False).head(k)
    top_rec_items = df.sort_values(by="probs", ascending=False).head(k)

    max_utility = sum(ratings)
    pred_utility = sum(top_rec_items.rating.values)

    return pred_utility / max_utility


def get_mean_utility_at_k(df: pd.DataFrame, k: int) -> float:
    return df.groupby("user_id").apply(user_utility_at_k, k).mean()


def plot_user_utility_per_group(test_df: pd.DataFrame, group_col: str, k: int) -> None:
    utility_per_group = defaultdict(list)
    for group, group_df in test_df.groupby(group_col):
        utility = get_mean_utility_at_k(group_df, k)
        utility_per_group[group] = utility

    plt.bar(utility_per_group.keys(), utility_per_group.values())
    plt.xlabel(" ".join(group_col.split("_")).capitalize())
    plt.xticks(rotation=90)
    plt.ylabel("User utility")
    plt.title(f"User utility {group_col} (higher is better)")
    plt.show()


def calculate_precision_recall(user_ratings, k, threshold):
    user_ratings.sort(key=lambda x: x[0], reverse=True)
    n_rel = sum(true_r >= threshold for _, true_r in user_ratings)
    n_rec_k = sum(est >= threshold for est, _ in user_ratings[:k])
    n_rel_and_rec_k = sum(
        (true_r >= threshold) and (est >= threshold) for est, true_r in user_ratings[:k]
    )

    precision = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
    recall = n_rel_and_rec_k / n_rel if n_rel != 0 else 1
    return precision, recall
