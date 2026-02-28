"""Potential functions V(phi) and their derivatives for the Klein-Gordon equation.

The KGE reads:  d²phi/dt² = d²phi/dx² - dV/dphi

Each potential is a pair of functions: (V, dV_dphi).
The solver only needs dV_dphi, but V is useful for energy calculations.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# phi^4 double-well potential:  V(phi) = (phi^2 - 1)^2
# ---------------------------------------------------------------------------

def phi4_potential(phi: np.ndarray) -> np.ndarray:
    """V(phi) = (phi^2 - 1)^2."""
    return (phi**2 - 1.0) ** 2


def phi4_derivative(phi: np.ndarray) -> np.ndarray:
    """dV/dphi = 4*phi*(phi^2 - 1)."""
    return 4.0 * phi * (phi**2 - 1.0)


# ---------------------------------------------------------------------------
# Sine-Gordon potential:  V(phi) = 1 - cos(phi)
# ---------------------------------------------------------------------------

def sine_gordon_potential(phi: np.ndarray) -> np.ndarray:
    """V(phi) = 1 - cos(phi)."""
    return 1.0 - np.cos(phi)


def sine_gordon_derivative(phi: np.ndarray) -> np.ndarray:
    """dV/dphi = sin(phi)."""
    return np.sin(phi)


# ---------------------------------------------------------------------------
# Free massive scalar field:  V(phi) = (m^2 / 2) * phi^2
# ---------------------------------------------------------------------------

def free_field_potential(phi: np.ndarray, m: float = 1.0) -> np.ndarray:
    """V(phi) = (m^2 / 2) * phi^2."""
    return 0.5 * m**2 * phi**2


def free_field_derivative(phi: np.ndarray, m: float = 1.0) -> np.ndarray:
    """dV/dphi = m^2 * phi."""
    return m**2 * phi
