"""RK4 time integrator for the 1D Klein-Gordon equation.

Solves:  d²phi/dt² = d²phi/dx² - dV/dphi(phi)

using the method of lines: spatial derivatives are discretised with second-order
central differences, and the resulting ODE system is advanced in time with the
classical four-stage Runge-Kutta method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from kg_solver.domain import Domain


def _laplacian(phi: np.ndarray, dx: float) -> np.ndarray:
    """Second-order central finite-difference Laplacian on interior points.

    Returns an array of the same length as *phi*, with zero at the boundaries.
    """
    lap = np.zeros_like(phi)
    lap[1:-1] = (phi[2:] - 2.0 * phi[1:-1] + phi[:-2]) / dx**2
    return lap


def _rhs(
    phi: np.ndarray,
    pi: np.ndarray,
    dx: float,
    dV: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Right-hand side of the first-order system.

    The KGE is rewritten as two first-order equations:
        dphi/dt = pi
        dpi/dt  = d²phi/dx² - dV/dphi

    Returns (dphi_dt, dpi_dt).
    """
    dphi_dt = pi.copy()
    dpi_dt = _laplacian(phi, dx) - dV(phi)
    # Enforce fixed (Dirichlet) boundaries: rates are zero there.
    dphi_dt[0] = dphi_dt[-1] = 0.0
    dpi_dt[0] = dpi_dt[-1] = 0.0
    return dphi_dt, dpi_dt


def solve(
    domain: Domain,
    phi_0: np.ndarray,
    dphi_dt_0: np.ndarray,
    dV: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the 1D Klein-Gordon equation with RK4 time stepping.

    Parameters
    ----------
    domain : Domain
        Grid specification (spatial and temporal).
    phi_0 : ndarray, shape (nx,)
        Initial field profile.
    dphi_dt_0 : ndarray, shape (nx,)
        Initial time derivative of the field.
    dV : callable
        Derivative of the potential, dV/dphi(phi).  Must accept and return
        a numpy array of the same shape as phi.

    Returns
    -------
    phi : ndarray, shape (nt+1, nx)
        Field values at every time step.
    pi : ndarray, shape (nt+1, nx)
        Time-derivative values at every time step.
    """
    dt = domain.dt
    dx = domain.dx
    nt = domain.nt
    nx = domain.nx

    # Allocate output arrays
    phi = np.zeros((nt + 1, nx))
    pi = np.zeros((nt + 1, nx))
    phi[0] = phi_0.copy()
    pi[0] = dphi_dt_0.copy()

    for n in range(nt):
        phi_n = phi[n]
        pi_n = pi[n]

        # Stage 1
        k1_phi, k1_pi = _rhs(phi_n, pi_n, dx, dV)

        # Stage 2
        k2_phi, k2_pi = _rhs(
            phi_n + 0.5 * dt * k1_phi,
            pi_n + 0.5 * dt * k1_pi,
            dx,
            dV,
        )

        # Stage 3
        k3_phi, k3_pi = _rhs(
            phi_n + 0.5 * dt * k2_phi,
            pi_n + 0.5 * dt * k2_pi,
            dx,
            dV,
        )

        # Stage 4
        k4_phi, k4_pi = _rhs(
            phi_n + dt * k3_phi,
            pi_n + dt * k3_pi,
            dx,
            dV,
        )

        # Update
        phi[n + 1] = phi_n + (dt / 6.0) * (k1_phi + 2 * k2_phi + 2 * k3_phi + k4_phi)
        pi[n + 1] = pi_n + (dt / 6.0) * (k1_pi + 2 * k2_pi + 2 * k3_pi + k4_pi)

    return phi, pi
