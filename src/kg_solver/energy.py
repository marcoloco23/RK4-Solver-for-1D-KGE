"""Energy diagnostics for the Klein-Gordon field.

The total energy (Hamiltonian) of the 1D Klein-Gordon field is:

    E = integral[ 1/2 (dphi/dt)^2 + 1/2 (dphi/dx)^2 + V(phi) ] dx

Energy should be conserved by the exact PDE; deviations measure numerical error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from kg_solver.domain import Domain


def energy_density(
    phi: np.ndarray,
    pi: np.ndarray,
    dx: float,
    V: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Energy density at each interior grid point for a single time slice.

    Parameters
    ----------
    phi : ndarray, shape (nx,)
    pi : ndarray, shape (nx,)  — dphi/dt
    dx : float
    V : callable — potential function V(phi)

    Returns
    -------
    rho : ndarray, shape (nx,)
        Energy density (zero at boundary points).
    """
    rho = np.zeros_like(phi)
    # Kinetic: 1/2 (dphi/dt)^2
    rho[1:-1] = 0.5 * pi[1:-1] ** 2
    # Gradient: 1/2 (dphi/dx)^2 using central differences
    grad_phi = (phi[2:] - phi[:-2]) / (2.0 * dx)
    rho[1:-1] += 0.5 * grad_phi**2
    # Potential
    rho[1:-1] += V(phi[1:-1])
    return rho


def total_energy(
    phi: np.ndarray,
    pi: np.ndarray,
    domain: Domain,
    V: Callable[[np.ndarray], np.ndarray],
) -> float:
    """Integrated total energy for a single time slice (trapezoidal rule)."""
    rho = energy_density(phi, pi, domain.dx, V)
    return float(np.sum(rho) * domain.dx)


def energy_timeseries(
    phi: np.ndarray,
    pi: np.ndarray,
    domain: Domain,
    V: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Total energy at each saved time step.

    Parameters
    ----------
    phi : ndarray, shape (nt+1, nx)
    pi : ndarray, shape (nt+1, nx)

    Returns
    -------
    E : ndarray, shape (nt+1,)
    """
    nt_plus_1 = phi.shape[0]
    E = np.empty(nt_plus_1)
    for n in range(nt_plus_1):
        E[n] = total_energy(phi[n], pi[n], domain, V)
    return E
