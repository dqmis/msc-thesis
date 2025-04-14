from collections import Counter, defaultdict

import numpy as np


def sample_utility(
    rel_matrix: np.ndarray, users_sample: int, items_sample: int
) -> np.ndarray:
    rng = np.random.default_rng()
    users_count, items_count = rel_matrix.shape

    users = rng.choice(users_count, size=users_sample, replace=False)
    items = rng.choice(items_count, size=items_sample, replace=False)
    return rel_matrix[users][:, items]


def sample_candidate_items(
    rel_matrix_users: np.ndarray, n_item_candidates: int
) -> np.ndarray:
    top_items = np.argsort(rel_matrix_users, axis=1)[:, -n_item_candidates:]
    unique_items = np.unique(top_items)
    return rel_matrix_users[np.arange(rel_matrix_users.shape[0])[:, None], unique_items]


def sample_users_from_groups(
    users_count: int,
    items_count: int,
    groups_map: dict[int, str],
    group_name: str,
    data: np.ndarray,
    naive_sampling: bool = True,
) -> tuple[np.ndarray, list[int]]:
    # Calculate the number of users to sample per group
    group_counts = Counter(row[group_name] for row in groups_map)
    users_per_group = {
        group: max(round(users_count * count / len(groups_map)), 1)
        for group, count in group_counts.items()
    }

    # Group users by their group
    group_users = defaultdict(list)
    for row in groups_map:
        group_users[row[group_name]].append(row["user_id"])

    # Sample users from each group
    sampled_users = [
        user
        for group, users in group_users.items()
        for user in np.random.choice(users, users_per_group[group], replace=False)
    ]

    rng = np.random.default_rng()
    if naive_sampling:
        items = rng.choice(data.shape[1], size=items_count, replace=False)
        return data[sampled_users][:, items], sampled_users

    rel_matrix_sampled = data[sampled_users]
    top_items = sample_candidate_items(rel_matrix_sampled, items_count)
    return top_items, sampled_users
