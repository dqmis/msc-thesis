import cvxpy as cp
import numpy as np


def compute_producer_optimal_solution(
    rel_matrix: np.ndarray, k_rec: int, solver=cp.SCIP
) -> tuple[float, np.ndarray]:
    """
    Given a relevance matrix, this function returns the convex optimization problem
    that maximizes the minimal producer utility given a fixed number of items to recommend.
    """
    allocations = cp.Variable(rel_matrix.shape, boolean=True)

    constraints = [
        # recommend k producers
        cp.sum(allocations, axis=1) == k_rec,
    ]

    # maximize the minimal item utility
    problem = cp.Problem(
        cp.Maximize(cp.min(cp.sum(allocations, axis=0))),
        constraints,
    )
    problem.solve(solver=solver)

    return problem.value, problem.variables()[0].value


def compute_consumer_optimal_solution(
    rel_matrix: np.ndarray,
    k_rec: int,
    producer_max_min_utility: float,
    gamma: float,
    method: str = "mean",
    solver: str = cp.SCIP,
    **kwargs: dict,
) -> cp.Problem:
    VALID_METHODS = ["mean", "min", "cvar"]
    if method not in VALID_METHODS:
        raise ValueError(f"Invalid method. Choose from {VALID_METHODS}")

    if method == "mean":
        return _compute_consumer_optimal_solution_mean(
            rel_matrix, k_rec, producer_max_min_utility, gamma, solver=solver
        )
    elif method == "min":
        return _compute_consumer_optimal_solution_min(
            rel_matrix, k_rec, producer_max_min_utility, gamma, solver=solver
        )
    elif method == "cvar":
        return _compute_consumer_optimal_solution_cvar(
            rel_matrix, k_rec, producer_max_min_utility, gamma, solver=solver, **kwargs
        )


def _compute_consumer_optimal_solution_min(
    rel_matrix: np.ndarray, k_rec: int, producer_max_min_utility: float, gamma: float, solver: str
) -> tuple[float, np.ndarray]:
    """
    Given a relevance matrix, this function returns the convex optimization problem
    that maximizes the minimal consumer utility given a fixed number of producers to recommend.
    """

    allocations = cp.Variable(rel_matrix.shape, boolean=True)
    # constraints
    constraints = [
        # recommend k producers
        cp.sum(allocations, axis=1) == k_rec,
        # minimal producer utility must be at least gamma * producer_max_min_utility
        cp.sum(allocations, axis=0) >= gamma * producer_max_min_utility,
    ]

    # maximize the mean consumer utility
    problem = cp.Problem(
        cp.Maximize(cp.min(cp.sum(cp.multiply(allocations, rel_matrix), axis=1))),
        constraints,
    )
    problem.solve(solver=cp.SCIP)

    return problem.value, problem.variables()[0].value


def _compute_consumer_optimal_solution_mean(
    rel_matrix: np.ndarray, k_rec: int, producer_max_min_utility: float, gamma: float, solver: str
) -> tuple[float, np.ndarray]:
    """
    Given a relevance matrix, this function returns the convex optimization problem
    that maximizes the mean consumer utility given a fixed number of producers to recommend.
    """

    allocations = cp.Variable(rel_matrix.shape, boolean=True)
    # constraints
    constraints = [
        # recommend k producers
        cp.sum(allocations, axis=1) == k_rec,
        # minimal producer utility must be at least gamma * producer_max_min_utility
        cp.sum(allocations, axis=0) >= gamma * producer_max_min_utility,
    ]

    # maximize the mean consumer utility
    problem = cp.Problem(
        cp.Maximize(cp.mean(cp.sum(cp.multiply(allocations, rel_matrix), axis=1))),
        constraints,
    )
    problem.solve(solver=solver)

    return problem.value, problem.variables()[0].value


def _compute_consumer_optimal_solution_cvar(
    rel_matrix: np.ndarray,
    k_rec: int,
    producer_max_min_utility: float,
    gamma: float,
    group_assignments: list[int],
    alpha: float,
    solver: str,
) -> tuple[float, np.ndarray]:
    """
    Given a relevance matrix, this function returns the convex optimization problem
    that minimizes the Conditional Value at Risk (CVaR) of the groups of consumer utility given a fixed number
    of producers to recommend.

    The problem is solved using CVaR (Conditional Value at Risk) approach.
    The CVaR is defined as the average of the worst-case losses, where the worst-case losses are defined
    as the losses that exceed a certain threshold (1 - alpha).
    """

    # producer allocations
    allocations = cp.Variable(rel_matrix.shape, boolean=True)

    constraints = [
        # there should be k_rec producers allocated to consumer
        cp.sum(allocations, axis=1) == k_rec,
        # each producer should get at least gamma * optimal producer utility
        cp.sum(allocations, axis=0) >= gamma * producer_max_min_utility,
    ]

    # greedy producer allocations for consumer
    greedy_allocations = np.sort(rel_matrix, axis=1)[:, -k_rec:].sum(axis=1)

    # precomputing values for later processing
    unique_groups, group_indices = np.unique(group_assignments, return_inverse=True)
    num_groups = len(unique_groups)
    group_masks = [group_indices == i for i in range(num_groups)]
    group_sizes = np.array([mask.sum() for mask in group_masks])

    allocations = cp.sum(cp.multiply(rel_matrix, allocations), axis=1)
    # Compute normalized losses for all groups simultaneously (vectorized)
    normalized_losses = []
    for mask, size in zip(group_masks, group_sizes):
        group_alloc = allocations[mask]
        greedy_group_alloc = greedy_allocations[mask]

        # compute loss for each group
        normalized_loss = cp.sum(1 - (group_alloc / greedy_group_alloc)) / size
        normalized_losses.append(normalized_loss)

    # CVaR computation (vectorized)
    rho = cp.Variable(nonneg=True)
    cvar_objective = rho + (1 / ((1 - alpha) * num_groups)) * cp.sum(
        cp.pos(cp.hstack(normalized_losses) - rho)
    )

    # Define and solve the optimization problem
    problem = cp.Problem(cp.Minimize(cvar_objective), constraints)
    problem.solve(solver=solver)

    return problem.value, problem.variables()[1].value
