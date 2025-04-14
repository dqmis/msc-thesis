import cvxpy as cp
import numpy as np


def best_unconstrained(U, k):
    # variables
    m, n = U.shape
    X_u = cp.Variable((m, n))

    # constraints
    constraints_u = []
    constraints_u.append(X_u @ np.ones(n) <= k * np.ones(m))
    constraints_u.append(X_u <= 1)
    constraints_u.append(X_u >= 0)

    problem_u = cp.Problem(cp.Maximize(cp.sum(cp.multiply(X_u, U))), constraints_u)
    problem_u.solve(solver="SCS")

    return problem_u.value


def constrained_maxmin_user_given_item(U, k, delta, v, u_star, just_value=True):
    m, n = U.shape

    X = cp.Variable((m, n))

    constraints = []

    constraints.append(X @ np.ones(n) <= k * np.ones(m))

    constraints.append(X <= 1)
    constraints.append(X >= 0)

    constraints.append(cp.sum(cp.multiply(X, U), axis=0) >= v * (U.T @ np.ones(m)))
    constraints.append(cp.sum(cp.multiply(X, U)) >= (1 - delta) * u_star)

    problem = cp.Problem(
        cp.Maximize(cp.min(cp.sum(cp.multiply(X, U), axis=1) / cp.max(U, axis=1))),
        constraints,
    )
    problem.solve(solver="SCS")

    maxmin = problem.value
    if just_value:
        return maxmin
    else:
        return problem


def constrained_maxmin_item_given_user(U, k, delta, v, u_star, just_value=True):
    m, n = U.shape

    X = cp.Variable((m, n))

    constraints = []

    constraints.append(X @ np.ones(n) <= k * np.ones(m))

    constraints.append(X <= 1)
    constraints.append(X >= 0)

    constraints.append(cp.sum(cp.multiply(X, U), axis=1) >= v * cp.max(U, axis=1))

    constraints.append(cp.sum(cp.multiply(X, U)) >= (1 - delta) * u_star)

    problem = cp.Problem(
        cp.Maximize(cp.min(cp.sum(cp.multiply(X, U), axis=0) / (U.T @ np.ones(m)))),
        constraints,
    )
    problem.solve(solver="SCS")

    maxmin = problem.value
    if just_value:
        return maxmin
    else:
        return problem


def price_of_fairness(U, k, delta):
    u_star = best_unconstrained(U, k)

    # unconstrained best user fairness:
    uf_opt = constrained_maxmin_user_given_item(U, k, delta, 0, u_star)

    if_opt = constrained_maxmin_item_given_user(U, k, delta, 0, u_star)
    uf_min = constrained_maxmin_user_given_item(U, k, delta, if_opt, u_star)

    pof = (uf_opt - uf_min) / uf_opt
    return pof, if_opt


def get_user_curve(U, k, delta, n_pts, just_value: bool = True) -> tuple[list, list]:
    # First, compute u* for the unconstrained problem
    u_star = best_unconstrained(U, k)

    # Now, do the actual convex optimization
    item_max = constrained_maxmin_item_given_user(U, k, delta, 0, u_star)
    user_max_problem = constrained_maxmin_user_given_item(U, k, delta, 0, u_star, False)
    user_max = user_max_problem.value
    u_max_vals = (U * user_max_problem.variables()[0].value).sum(axis=1)

    gamma_items = np.linspace(0.0, 1, n_pts)
    pairs = []
    u_util = []
    for gamma_item in gamma_items:
        v_user_problem = constrained_maxmin_user_given_item(
            U, k, delta, gamma_item * item_max, u_star, just_value=False
        )
        u_util.append(
            (U * v_user_problem.variables()[0].value).sum(axis=1) / u_max_vals
        )
        v_user = v_user_problem.value
        pair = (gamma_item, v_user / user_max) if just_value else (gamma_item, v_user)
        pairs.append(pair)

    return pairs
