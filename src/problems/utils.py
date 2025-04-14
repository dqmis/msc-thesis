from collections import Counter, defaultdict
from typing import Any

import numpy as np


def sample_data_for_group(
    n_consumers: int,
    n_producers: int,
    groups_map: list[dict[str, Any]],
    group_key: str,
    data: np.ndarray,
    naive_sampling: bool = True,
) -> tuple[np.ndarray, list[int], list[int]]:
    """
    Sample users from groups and return the sampled data, consumer ids, and group assignments.
    If naive_sampling is True, it samples random producer indices.
    Otherwise, it samples the top producers for each consumer.
    """

    initial_allocation = _compute_group_allocations(n_consumers, groups_map, group_key)

    group_consumers = defaultdict(list)
    for row in groups_map:
        group_consumers[row[group_key]].append(row["user_id"])

    # Sample users from each group according to the final allocation.
    consumers_ids = []
    for group, users in group_consumers.items():
        consumers_ids.extend(np.random.choice(users, initial_allocation[group], replace=False))

    # Use your helper to parse the groups of the sampled users.
    group_assignments = _parse_groups_ids(consumers_ids, groups_map, group_key)

    rng = np.random.default_rng()
    if naive_sampling:
        # Sample random item indices.
        sampled_producers = rng.choice(data.shape[1], size=n_producers, replace=False)
        return data[consumers_ids][:, sampled_producers], consumers_ids, group_assignments

    sampled_data = _sample_candidate_producers(data[consumers_ids], n_producers)
    return sampled_data, consumers_ids, group_assignments


def _sample_candidate_producers(data: np.ndarray, n_candidates: int) -> np.ndarray:
    """
    Sample a subset top producers for each consumer.
    """

    top_producers = np.argsort(data, axis=1)[:, -n_candidates:]
    distinct_producers = np.unique(top_producers)
    return data[np.arange(data.shape[0])[:, None], distinct_producers]


def _parse_groups_ids(consumer_ids: np.ndarray, groups_map: list[dict[str, int]], group_key: str):
    """Parses given group key value to int for each consumer"""

    consumer_groups = [i[group_key] for i in groups_map if i["user_id"] in consumer_ids]
    all_groups = list(set(consumer_groups))

    return np.array([all_groups.index(i) for i in consumer_groups])


def _compute_group_allocations(
    n_consumers: int, groups_map: list[dict[str, int]], group_key: str
) -> dict[str, int]:
    """
    Compute the number of users to sample from each group. It always samples the given number of users. If after
    initial allocation the total number of users is not equal to n_consumers, it will adjust the allocation
    by adding or removing users from the group with the most occurrences.
    """
    # Count how many times each group appears.
    group_freq = Counter(row[group_key] for row in groups_map)

    # Calculate initial allocation using proportions.
    # (Total items in groups_map is used as denominator to reflect each group's share.)
    initial_allocation = {
        group: max(round(n_consumers * count / len(groups_map)), 1) for group, count in group_freq.items()
    }

    # Adjust allocation so that sum equals exactly users_count.
    allocated_total = sum(initial_allocation.values())
    # Identify the "biggest" group (with the most occurrences)
    biggest_group = max(group_freq.items(), key=lambda x: x[1])[0]

    if allocated_total > n_consumers:
        # Too many allocated; remove the difference from the largest group, but keep at least one user.
        diff = allocated_total - n_consumers
        initial_allocation[biggest_group] = max(initial_allocation[biggest_group] - diff, 1)
    elif allocated_total < n_consumers:
        # Too few allocated; add the shortfall to the largest group.
        diff = n_consumers - allocated_total
        initial_allocation[biggest_group] += diff

    return initial_allocation
