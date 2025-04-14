from collections import defaultdict
from typing import Any

import numpy as np
from tqdm import tqdm

from src.problems.problems import compute_consumer_optimal_solution, compute_producer_optimal_solution
from src.problems.utils import sample_data_for_group
from src.utils import random_seed

ConsumerResult = tuple[list[float], list[float]]


def compute_consumer_producer_utils(
    rel_matrix: np.ndarray,
    producer_max_min_utility: float,
    gamma: float,
    k_rec: int,
    solver: str,
    method: str = "mean",
    method_kwargs: dict[str, Any] | None = None,
) -> tuple[list[float], list[float]]:
    greedy_allocations_per_consumer = np.sort(rel_matrix, axis=1)[:, -k_rec:].sum(axis=1)

    _, producer_assignments = compute_consumer_optimal_solution(
        rel_matrix,
        k_rec,
        gamma,
        producer_max_min_utility,
        method=method,
        solver=solver,
        **(method_kwargs or {}),
    )
    consumers_utils = (rel_matrix * producer_assignments).sum(axis=1) / greedy_allocations_per_consumer
    producers_utils = producer_assignments.sum(axis=0)

    return consumers_utils, producers_utils


def compute_utils_per_params(
    rel_matrix: np.ndarray,
    n_consumers: int,
    n_producers: int,
    k_rec: int,
    gammas: list[float],
    groups_map: list[dict[str, Any]],
    group_keys: list[str],
    solver: str,
    alphas: list[float] | None = None,
    use_naive_sampling: bool = True,
    method: str = "mean",
    n_runs: int = 1,
) -> dict[str, dict[str, dict[str, list[ConsumerResult]]]]:
    random_seed(42)

    if alphas is None:
        if method == "cvar":
            raise ValueError("Alpha must be provided for CVaR method.")
        alphas = [0.0]

    results = {}
    for group_key in group_keys:
        group_results = defaultdict(lambda: defaultdict(list[ConsumerResult]))
        for _ in tqdm(range(n_runs), position=0, leave=True, desc="Runs"):
            rel_matrix_sampled, consumer_ids, group_assignments = sample_data_for_group(
                n_consumers=n_consumers,
                n_producers=n_producers,
                groups_map=groups_map,
                group_key=group_key,
                data=rel_matrix,
                naive_sampling=use_naive_sampling,
            )

            producer_max_min_utility, _ = compute_producer_optimal_solution(
                rel_matrix=rel_matrix_sampled,
                k_rec=k_rec,
                solver=solver,
            )

            for alpha in tqdm(alphas, position=1, leave=True, desc="Alphas"):
                for gamma in tqdm(gammas, position=2, leave=True, desc="Gammas"):
                    _consumers_utils, _ = compute_consumer_producer_utils(
                        rel_matrix_sampled,
                        producer_max_min_utility,
                        gamma,
                        k_rec,
                        method=method,
                        solver=solver,
                        method_kwargs={
                            "alpha": alpha,
                            "group_assignments": group_assignments,
                        },
                    )
                    group_results[alpha][gamma].append((_consumers_utils, consumer_ids))
        results[group_key] = _parse_results_per_group_value(group_results, group_key, groups_map)

    return results


def _parse_results_per_group_value(
    results: dict[str, dict[str, list[ConsumerResult]]],
    group_key: str,
    groups_map: list[dict[str, Any]],
):
    group_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for alpha, alpha_results in results.items():
        for gamma, gamma_results in alpha_results.items():
            group_results["all"][alpha][gamma] = [[] for _ in range(len(gamma_results))]
            for run_id, run_results in enumerate(gamma_results):
                run_consumers_utils, run_consumers_ids = run_results
                run_consumer_groups = [i[group_key] for i in groups_map if i["user_id"] in run_consumers_ids]
                for distinct_group in set(run_consumer_groups):
                    run_consumer_groups_idx = np.where(np.array(run_consumer_groups) == distinct_group)[0]
                    group_results[distinct_group][alpha][gamma].append(
                        run_consumers_utils[run_consumer_groups_idx]
                    )
                    group_results["all"][alpha][gamma][run_id].extend(
                        run_consumers_utils[run_consumer_groups_idx]
                    )

    return group_results
