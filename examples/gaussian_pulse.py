#!/usr/bin/env python3
"""Gaussian pulse evolution in the phi^4 Klein-Gordon equation.

A simple getting-started example: watch a Gaussian pulse radiate into
dispersive waves in the phi^4 potential.
"""

import matplotlib.pyplot as plt
import numpy as np

from kg_solver.domain import Domain
from kg_solver.energy import energy_timeseries
from kg_solver.initial_conditions import gaussian
from kg_solver.potentials import phi4_derivative, phi4_potential
from kg_solver.solver import solve
from kg_solver.visualization import plot_snapshot, plot_spacetime

# --- Setup ---
domain = Domain(x_min=-10, x_max=10, nx=201, t_max=8.0, nt=1600)

print(f"Grid: {domain.nx} points, {domain.nt} steps")
print(f"CFL = {domain.cfl:.3f}")

# --- Solve ---
phi_0, dphi_dt_0 = gaussian(domain, amplitude=0.5, sigma=1.0)
phi, pi = solve(domain, phi_0, dphi_dt_0, phi4_derivative)

# --- Energy ---
E = energy_timeseries(phi, pi, domain, phi4_potential)
print(f"Energy drift: {abs(E[-1] - E[0]) / E[0]:.2e}")

# --- Plots ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

plot_snapshot(phi, domain, t_index=0, ax=axes[0], label="t = 0")
plot_snapshot(phi, domain, t_index=len(phi) // 2, ax=axes[0], label="t = T/2", color="blue")
plot_snapshot(phi, domain, t_index=-1, ax=axes[0], label="t = T", color="green")
axes[0].set_title("Snapshots")

plot_spacetime(phi, domain, ax=axes[1])
axes[1].set_title("Spacetime")

axes[2].plot(domain.t, E)
axes[2].set_xlabel("t")
axes[2].set_ylabel("E(t)")
axes[2].set_title("Energy conservation")

plt.suptitle(r"Gaussian pulse in $\phi^4$ potential", fontsize=14)
plt.tight_layout()
plt.savefig("gaussian_pulse.png", dpi=150)
plt.show()
