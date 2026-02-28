"""Grid and domain configuration for the 1D Klein-Gordon solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Domain:
    """Immutable specification of the spatial and temporal discretization.

    Parameters
    ----------
    x_min, x_max : float
        Spatial domain boundaries.
    nx : int
        Number of spatial grid points (including boundaries).
    t_max : float
        Final simulation time (starting from t=0).
    nt : int
        Number of time steps.
    """

    x_min: float
    x_max: float
    nx: int
    t_max: float
    nt: int

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / (self.nx - 1)

    @property
    def dt(self) -> float:
        return self.t_max / self.nt

    @property
    def x(self) -> np.ndarray:
        return np.linspace(self.x_min, self.x_max, self.nx)

    @property
    def t(self) -> np.ndarray:
        return np.linspace(0, self.t_max, self.nt + 1)

    @property
    def cfl(self) -> float:
        """CFL number (dt/dx). Should be < 1 for stability."""
        return self.dt / self.dx

    def refine(self, spatial_factor: int = 2, temporal_factor: int = 2) -> Domain:
        """Return a new domain with finer resolution."""
        return Domain(
            x_min=self.x_min,
            x_max=self.x_max,
            nx=(self.nx - 1) * spatial_factor + 1,
            t_max=self.t_max,
            nt=self.nt * temporal_factor,
        )
