from enum import Enum

import numpy as np

from src.optimizers.optimizers import (
    constrained_maxmin_item_given_user,
    constrained_maxmin_user_given_item,
    solve_maxmin_user_given_item,
)
from src.optimizers.utils import sample_users_from_groups, sample_utility


class SamplingType(Enum):
    DEFAULT = 1
    GROUPS = 2


def get_user_curve_for_gammas(
    rel_matrix: np.ndarray, gamma_points: list[float], k_rec: int
) -> tuple[list[tuple[float, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    item_max_min_v = constrained_maxmin_item_given_user(rel_matrix, k_rec).value
    user_max_min_result = constrained_maxmin_user_given_item(rel_matrix, k_rec, 0)
    user_max_min_v = user_max_min_result.value
    user_max_min_user_level = (user_max_min_result.variables()[0].value * rel_matrix).sum(axis=1)

    user_max_min_results = []
    users_v_values = []
    for gamma_item in gamma_points:
        _user_max_min_result, users_v = solve_maxmin_user_given_item(
            rel_matrix, item_max_min_v, k_rec, gamma_item
        )
        user_max_min_results.append((_user_max_min_result, user_max_min_v))
        users_v_values.append((users_v.sum(axis=1), user_max_min_user_level))

    return user_max_min_results, users_v_values


def get_user_curve(
    rel_matrix: np.ndarray,
    k_rec: int,
    gamma_points: list[float],
    n_runs: int = 10,
    sampling_type: SamplingType = SamplingType.DEFAULT,
    sampling_group_name: str | None = None,
    users_sample: int = 100,
    items_sample: int = 100,
    use_naive_sampling: bool = True,
) -> tuple[
    list[list[tuple[list[tuple[float, float]], list[tuple[float, float]]]]],
    list[list[np.ndarray]],
    list[list[np.ndarray]],
]:
    sampled_users_list = []
    user_max_min_results_list = []
    users_v_values_list = []
    for _ in range(n_runs):
        if sampling_type == SamplingType.DEFAULT:
            rel_matrix_sampled = sample_utility(rel_matrix, users_sample, items_sample)
        else:
            rel_matrix_sampled, sampled_users = sample_users_from_groups(
                rel_matrix.shape[0], rel_matrix.shape[1], sampling_group_name, rel_matrix, use_naive_sampling
            )
            sampled_users_list.append(sampled_users)

        user_max_min_results, users_v_values = get_user_curve_for_gammas(
            rel_matrix_sampled, gamma_points, k_rec
        )
        user_max_min_results_list.append(user_max_min_results)
        users_v_values_list.append(users_v_values)

    return user_max_min_results_list, users_v_values_list, sampled_users_list
