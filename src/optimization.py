import numpy as np
import cvxpy as cp

def equal_weight(stocks):
    """Compute equal-weight (1/N) allocation.

    Parameters
    ----------
    stocks : list[str]
        List of ticker symbols.

    Returns
    -------
    np.ndarray
        (N,) array where each weight equals 1/N.
    """
    weights = np.full(len(stocks), 1/len(stocks))
    return weights

def min_variance_portfolio(cov: np.ndarray, max_weight: float | None = None):
    """Solve the long-only minimum variance portfolio using CVXPY.

    Parameters
    ----------
    cov : np.ndarray
        Annualized N×N covariance matrix (must be symmetric PSD).
    max_weight : float or None
        Upper bound on each individual weight (e.g. 0.40). None = no cap.

    Returns
    -------
    np.ndarray
        (N,) optimal weight vector summing to 1.

    Raises
    ------
    ValueError
        If CVXPY finds no feasible solution.
    """
    cov = np.asarray(cov)

    # Safety: enforce symmetry (helps numerical stability)
    cov = 0.5 * (cov + cov.T)

    # Calculate the number of stocks
    n = cov.shape[0]

    # We define the number of variables
    w = cp.Variable(n)

    # We define what we need to minimize
    objective = cp.Minimize(cp.quad_form(w, cov))

    # We define the constraints
    constraints = [
        cp.sum(w) == 1,  # The sum of all weights should be 1
        w >= 0           # Long-only
    ]

    # Add max weight cap if provided
    if max_weight is not None:
        constraints.append(w <= max_weight)

    # We define the problem to solve
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if w.value is None:
        raise ValueError("Optimization failed (no solution found).")

    return w.value


def max_sharpe_ratio(mu: np.ndarray, cov: np.ndarray, rf: float = 0.0, max_weight: float | None = None):
    """Solve the long-only maximum Sharpe ratio portfolio using CVXPY.

    Uses the convex reformulation: maximize excess return subject to
    quadratic risk constraint, then normalize weights to sum to 1.

    Parameters
    ----------
    mu : np.ndarray
        (N,) annualized expected return vector.
    cov : np.ndarray
        Annualized N×N covariance matrix (symmetric PSD).
    rf : float
        Annual risk-free rate (e.g. 0.02 for 2%).
    max_weight : float or None
        Upper bound per asset. None = no cap.

    Returns
    -------
    np.ndarray
        (N,) optimal weight vector summing to 1.

    Raises
    ------
    ValueError
        If CVXPY finds no feasible solution.
    """
    mu = np.asarray(mu).flatten()
    cov = np.asarray(cov)

    # Safety: enforce symmetry
    cov = 0.5 * (cov + cov.T)

    n = len(mu)
    w = cp.Variable(n)

    excess_return = mu - rf

    objective = cp.Maximize(cp.sum(cp.multiply(excess_return, w)))

    constraints = [
        cp.quad_form(w, cov) <= 1,
        w >= 0
    ]

    # Add max weight cap if provided
    if max_weight is not None:
        constraints.append(w <= max_weight)

    problem = cp.Problem(objective, constraints)
    problem.solve()

    if w.value is None:
        raise ValueError("Optimization failed (no solution found).")

    # Normalize weights to sum to 1
    w_raw = w.value
    w_norm = w_raw / np.sum(w_raw)

    return w_norm


def efficient_frontier(mu, cov, n_points=30, long_only=True, max_weight: float | None = None):
    """Compute points on the mean-variance efficient frontier.

    For each target return between min(mu) and max(mu), solves a
    minimum-variance problem with a return floor constraint.

    Parameters
    ----------
    mu : np.ndarray
        (N,) annualized expected return vector.
    cov : np.ndarray
        Annualized N×N covariance matrix (symmetric PSD).
    n_points : int
        Number of points to compute on the frontier.
    long_only : bool
        If True, enforce w >= 0 constraint.
    max_weight : float or None
        Upper bound per asset. None = no cap.

    Returns
    -------
    vols : np.ndarray
        Annualized volatilities for each frontier point.
    rets : np.ndarray
        Annualized returns for each frontier point.
    weights_list : list[np.ndarray]
        Optimal weight vectors for each point.
    """
    mu = np.asarray(mu).flatten()
    cov = np.asarray(cov)

    # Safety: enforce symmetry
    cov = 0.5 * (cov + cov.T)

    n = len(mu)

    # Grid of target returns
    r_min, r_max = float(mu.min()), float(mu.max())
    targets = np.linspace(r_min, r_max, n_points)

    vols = []
    rets = []
    weights_list = []

    for R in targets:
        w = cp.Variable(n)
        risk = cp.quad_form(w, cov)

        # Use cvxpy-friendly return expression
        port_ret_expr = cp.sum(cp.multiply(mu, w))

        constraints = [cp.sum(w) == 1]

        if long_only:
            constraints.append(w >= 0)

        if max_weight is not None:
            constraints.append(w <= max_weight)

        constraints.append(port_ret_expr >= R)

        prob = cp.Problem(cp.Minimize(risk), constraints)
        prob.solve()

        if w.value is None:
            continue

        w_opt = w.value
        port_ret = float(mu @ w_opt)
        port_vol = float(np.sqrt(w_opt.T @ cov @ w_opt))

        rets.append(port_ret)
        vols.append(port_vol)
        weights_list.append(w_opt)

    return np.array(vols), np.array(rets), weights_list