import cvxpy as cp
import numpy as np
import tqdm as tqdm


def constrained_maxmin_user_given_item(rel_matrix: np.ndarray, k_rec: int = 1, v: float = 0) -> cp.Problem:
    """
    Given a relevance matrix, this function returns the convex optimization problem
    that maximizes the minimal user utility given a fixed number of items to recommend
    and a fixed minimal item utility.

    If v which represents item_value_star * gamma is set to 0, the problem will maximize
    unconstrained user utility.
    """
    x_alloc = cp.Variable(rel_matrix.shape, boolean=True)

    # constraints
    constraints = [
        # recommend k items
        cp.sum(x_alloc, axis=1) == k_rec,
        # minimal item utility must be at least v
        cp.min(cp.sum(x_alloc, axis=0)) >= v,
    ]

    # maximize the minimal user utility
    problem = cp.Problem(
        cp.Maximize(cp.min(cp.sum(cp.multiply(x_alloc, rel_matrix), axis=1))),
        constraints,
    )
    problem.solve(solver=cp.SCIP)

    return problem


def constrained_maxmin_item_given_user(rel_matrix: np.ndarray, k_rec: int = 1) -> cp.Problem:
    """
    Given a relevance matrix, this function returns the convex optimization problem
    that maximizes the minimal item utility given a fixed number of items to recommend.
    """
    x_alloc = cp.Variable(rel_matrix.shape, boolean=True)

    # constraints
    constraints = [
        # recommend k items
        cp.sum(x_alloc, axis=1) == k_rec,
    ]

    # maximize the minimal item utility
    problem = cp.Problem(
        cp.Maximize(cp.min(cp.sum(x_alloc, axis=0))),
        constraints,
    )
    problem.solve(solver=cp.SCIP)

    return problem


def solve_maxmin_user_given_item(
    rel_matrix: np.ndarray, item_max_min_v: float | None = None, k_rec: int = 1, gamma: float = 0
) -> tuple[float, np.ndarray]:
    """
    Given a relevance matrix, this function solves the convex optimization problem
    that maximizes the minimal user utility given a fixed number of items to recommend
    and a fixed minimal item utility.
    """
    # if gamma is set to 0, the problem will maximize unconstrained user utility
    item_max_min_v = item_max_min_v or (
        constrained_maxmin_item_given_user(rel_matrix, k_rec).value if gamma > 0 else 0
    )
    user_max_min_result = constrained_maxmin_user_given_item(rel_matrix, k_rec, gamma * item_max_min_v)
    # compute the user utility vector using the optimal allocation of items
    users_v = user_max_min_result.variables()[0].value * rel_matrix

    return user_max_min_result.value, users_v
