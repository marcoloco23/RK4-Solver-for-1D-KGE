"""Initial condition factories for the 1D Klein-Gordon equation.

Each function returns (phi_0, dphi_dt_0) — the initial field profile and its
time derivative — as 1-D numpy arrays on the spatial grid defined by a Domain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from kg_solver.domain import Domain

SQRT2 = np.sqrt(2.0)


def gaussian(domain: Domain, amplitude: float = 1.0, sigma: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian pulse centred at the origin, initially at rest.

    phi_0(x) = amplitude * exp(-x^2 / (2*sigma^2))
    dphi_dt_0 = 0
    """
    x = domain.x
    phi_0 = amplitude * np.exp(-x**2 / (2.0 * sigma**2))
    # Enforce Dirichlet boundaries
    phi_0[0] = 0.0
    phi_0[-1] = 0.0
    dphi_dt_0 = np.zeros_like(x)
    return phi_0, dphi_dt_0


def sine_mode(domain: Domain, n: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Standing sine mode phi_0 = sin(n*pi*x/L), initially at rest."""
    x = domain.x
    L = domain.x_max - domain.x_min
    phi_0 = np.sin(n * np.pi * (x - domain.x_min) / L)
    dphi_dt_0 = np.zeros_like(x)
    return phi_0, dphi_dt_0


def vacuum(domain: Domain, sign: float = -1.0) -> tuple[np.ndarray, np.ndarray]:
    """Uniform vacuum state phi = +/-1 for the phi^4 potential."""
    x = domain.x
    phi_0 = np.full_like(x, sign)
    dphi_dt_0 = np.zeros_like(x)
    return phi_0, dphi_dt_0


def static_kink(domain: Domain) -> tuple[np.ndarray, np.ndarray]:
    """Static phi^4 kink centred at the origin.

    phi_0(x) = tanh(x / sqrt(2))
    """
    x = domain.x
    phi_0 = np.tanh(SQRT2 * x)
    dphi_dt_0 = np.zeros_like(x)
    return phi_0, dphi_dt_0


def boosted_kink(
    domain: Domain, beta: float, x0: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Lorentz-boosted phi^4 kink.

    Parameters
    ----------
    beta : float
        Velocity in units of c (|beta| < 1).
    x0 : float
        Centre of the kink at t=0.
    """
    x = domain.x
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    xi = SQRT2 * gamma * (x - x0)  # boosted coordinate at t=0
    phi_0 = np.tanh(xi)
    dphi_dt_0 = -SQRT2 * gamma * beta / np.cosh(xi) ** 2
    return phi_0, dphi_dt_0


def kink_antikink_collision(
    domain: Domain, beta: float
) -> tuple[np.ndarray, np.ndarray]:
    """Superposition of a kink (moving right) and an anti-kink (moving left).

    They are placed at x = -L/4 and x = +L/4 respectively, each with speed
    |beta| towards the centre. Uses linear superposition as an approximate IC.

    Parameters
    ----------
    beta : float
        Speed of each soliton towards the centre (positive value).
    """
    L = domain.x_max - domain.x_min
    x0 = L / 4.0
    # Kink at -L/4 moving right (positive beta)
    phi_k, dphi_k = boosted_kink(domain, beta=abs(beta), x0=-x0)
    # Anti-kink at +L/4 moving left: tanh(-xi) = -tanh(xi), negate spatial arg
    phi_ak, dphi_ak = _boosted_antikink(domain, beta=abs(beta), x0=x0)
    return phi_k + phi_ak, dphi_k + dphi_ak


def _boosted_antikink(
    domain: Domain, beta: float, x0: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Lorentz-boosted anti-kink (spatial reflection of kink) moving left."""
    x = domain.x
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    xi = SQRT2 * gamma * (-(x - x0))  # note the negation for anti-kink
    phi_0 = np.tanh(xi)
    dphi_dt_0 = SQRT2 * gamma * beta / np.cosh(xi) ** 2  # sign flipped vs kink
    return phi_0, dphi_dt_0


def exact_kink_antikink(
    domain: Domain, beta: float
) -> np.ndarray:
    """Exact (linear superposition) solution for kink-antikink collision.

    Returns phi[nt+1, nx] — the full spacetime field for comparison with
    the numerical solution. Note: this is only exact for well-separated
    solitons; it serves as a reference for convergence testing.
    """
    x = domain.x
    t = domain.t
    L = domain.x_max - domain.x_min
    x0 = L / 4.0
    gamma = 1.0 / np.sqrt(1.0 - beta**2)

    # Broadcast: x[nx], t[nt+1] -> phi[nt+1, nx]
    X = x[np.newaxis, :]  # (1, nx)
    T = t[:, np.newaxis]  # (nt+1, 1)

    xi_kink = SQRT2 * gamma * ((X + x0) - abs(beta) * T)
    xi_antikink = SQRT2 * gamma * (-(X - x0) - abs(beta) * T)

    return np.tanh(xi_kink) + np.tanh(xi_antikink)
