"""Plotting and animation utilities for Klein-Gordon solutions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from kg_solver.domain import Domain


def plot_snapshot(
    phi: np.ndarray,
    domain: Domain,
    t_index: int = 0,
    ax: plt.Axes | None = None,
    label: str | None = None,
    **kwargs,
) -> Figure:
    """Plot the field at a single time step."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(domain.x, phi[t_index], label=label, **kwargs)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\phi$")
    ax.set_ylim(-2.5, 2.5)
    if label:
        ax.legend()
    fig.tight_layout()
    return fig


def plot_spacetime(
    phi: np.ndarray,
    domain: Domain,
    ax: plt.Axes | None = None,
    **kwargs,
) -> Figure:
    """Pseudocolour plot of phi(x, t)."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    extent = [domain.x_min, domain.x_max, 0, domain.t_max]
    defaults = dict(aspect="auto", origin="lower", cmap="RdBu_r", vmin=-2, vmax=2)
    defaults.update(kwargs)
    im = ax.imshow(phi, extent=extent, **defaults)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    fig.colorbar(im, ax=ax, label=r"$\phi$")
    fig.tight_layout()
    return fig


def animate_solution(
    phi: np.ndarray,
    domain: Domain,
    interval: int = 30,
    stride: int = 1,
) -> FuncAnimation:
    """Create a matplotlib animation of the time evolution."""
    fig, ax = plt.subplots()
    ax.set_xlim(domain.x_min, domain.x_max)
    ax.set_ylim(-2.5, 2.5)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\phi$")
    (line,) = ax.plot([], [], lw=2)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    frames = range(0, phi.shape[0], stride)
    t = domain.t

    def update(n):
        line.set_data(domain.x, phi[n])
        time_text.set_text(f"t = {t[n]:.3f}")
        return line, time_text

    return FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)


def plot_energy(
    energy: np.ndarray,
    domain: Domain,
    ax: plt.Axes | None = None,
) -> Figure:
    """Plot total energy vs. time."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(domain.t, energy)
    ax.set_xlabel("t")
    ax.set_ylabel("E(t)")
    ax.set_title("Energy conservation")
    fig.tight_layout()
    return fig


def plot_convergence(
    grid_sizes: list[int],
    errors: list[float],
    ax: plt.Axes | None = None,
) -> Figure:
    """Log-log plot of error vs. grid spacing."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    dx_values = [1.0 / (n - 1) for n in grid_sizes]
    ax.loglog(dx_values, errors, "o-", label="L2 error")
    # Reference 4th-order slope
    dx_arr = np.array(dx_values)
    ref = errors[0] * (dx_arr / dx_arr[0]) ** 4
    ax.loglog(dx_arr, ref, "--", alpha=0.5, label=r"$\mathcal{O}(\Delta x^4)$")
    ax.set_xlabel(r"$\Delta x$")
    ax.set_ylabel("L2 error")
    ax.legend()
    ax.set_title("Grid convergence")
    fig.tight_layout()
    return fig
