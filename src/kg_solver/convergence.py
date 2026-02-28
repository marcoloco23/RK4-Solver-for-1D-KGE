"""Grid convergence and error analysis for the Klein-Gordon solver."""

from __future__ import annotations

from typing import Callable

import numpy as np

from kg_solver.domain import Domain
from kg_solver.solver import solve


def l2_error(numerical: np.ndarray, exact: np.ndarray, dx: float) -> float:
    """Discrete L2 norm of the error: sqrt( dx * sum((num - exact)^2) )."""
    return float(np.sqrt(dx * np.sum((numerical - exact) ** 2)))


def l2_error_timeseries(
    phi_num: np.ndarray,
    phi_exact: np.ndarray,
    dx: float,
) -> np.ndarray:
    """L2 error at each time step.

    Parameters
    ----------
    phi_num, phi_exact : ndarray, shape (nt+1, nx)

    Returns
    -------
    errors : ndarray, shape (nt+1,)
    """
    diff = phi_num - phi_exact
    return np.sqrt(dx * np.sum(diff**2, axis=1))


def convergence_order(
    error_coarse: float, error_fine: float, refinement_factor: int = 2
) -> float:
    """Estimated convergence order from two grid levels.

    order = log2(error_coarse / error_fine) / log2(refinement_factor)
    """
    if error_fine == 0 or error_coarse == 0:
        return float("inf")
    return float(np.log2(error_coarse / error_fine) / np.log2(refinement_factor))


def convergence_study(
    base_domain: Domain,
    ic_factory: Callable[[Domain], tuple[np.ndarray, np.ndarray]],
    dV: Callable[[np.ndarray], np.ndarray],
    exact_solution: Callable[[Domain], np.ndarray],
    refinements: int = 3,
) -> dict:
    """Run a grid-refinement convergence study.

    Parameters
    ----------
    base_domain : Domain
        Coarsest grid.
    ic_factory : callable
        Returns (phi_0, dphi_dt_0) given a Domain.
    dV : callable
        Potential derivative.
    exact_solution : callable
        Returns phi_exact[nt+1, nx] given a Domain.
    refinements : int
        Number of successive 2x refinements.

    Returns
    -------
    dict with keys:
        "grid_sizes" : list of nx values
        "errors" : list of L2 errors at final time
        "orders" : list of estimated convergence orders
    """
    domains = [base_domain]
    for _ in range(refinements):
        domains.append(domains[-1].refine())

    errors = []
    grid_sizes = []

    for dom in domains:
        phi_0, dphi_dt_0 = ic_factory(dom)
        phi_num, _ = solve(dom, phi_0, dphi_dt_0, dV)
        phi_ex = exact_solution(dom)
        err = l2_error(phi_num[-1], phi_ex[-1], dom.dx)
        errors.append(err)
        grid_sizes.append(dom.nx)

    orders = [
        convergence_order(errors[i], errors[i + 1]) for i in range(len(errors) - 1)
    ]

    return {"grid_sizes": grid_sizes, "errors": errors, "orders": orders}
