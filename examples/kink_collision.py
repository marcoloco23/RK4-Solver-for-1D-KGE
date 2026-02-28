#!/usr/bin/env python3
"""Kink-antikink collision in the phi^4 Klein-Gordon equation.

This reproduces the main result from the original notebook: two solitons
colliding and scattering, compared against an exact superposition reference.
"""

import matplotlib.pyplot as plt
import numpy as np

from kg_solver.convergence import l2_error_timeseries
from kg_solver.domain import Domain
from kg_solver.energy import energy_timeseries
from kg_solver.initial_conditions import exact_kink_antikink, kink_antikink_collision
from kg_solver.potentials import phi4_derivative, phi4_potential
from kg_solver.solver import solve
from kg_solver.visualization import plot_energy, plot_spacetime

# --- Setup ---
domain = Domain(x_min=-np.pi, x_max=np.pi, nx=161, t_max=3 * np.pi, nt=3200)
beta = 0.5

print(f"Grid: {domain.nx} points, {domain.nt} steps")
print(f"dx = {domain.dx:.4f}, dt = {domain.dt:.4f}, CFL = {domain.cfl:.3f}")

# --- Solve ---
phi_0, dphi_dt_0 = kink_antikink_collision(domain, beta=beta)
phi, pi = solve(domain, phi_0, dphi_dt_0, phi4_derivative)

# --- Compare to exact superposition ---
phi_exact = exact_kink_antikink(domain, beta=beta)
errors = l2_error_timeseries(phi, phi_exact, domain.dx)
print(f"L2 error at final time: {errors[-1]:.4e}")

# --- Energy conservation ---
E = energy_timeseries(phi, pi, domain, phi4_potential)
print(f"Energy drift: {abs(E[-1] - E[0]) / E[0]:.2e}")

# --- Visualise ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Spacetime plot
plot_spacetime(phi, domain, ax=axes[0])
axes[0].set_title("Numerical solution")

# Spacetime exact
plot_spacetime(phi_exact, domain, ax=axes[1])
axes[1].set_title("Exact superposition")

# Energy
plot_energy(E, domain, ax=axes[2])

plt.suptitle(rf"Kink-antikink collision ($\beta = {beta}$)", fontsize=14)
plt.tight_layout()
plt.savefig("kink_collision.png", dpi=150)
plt.show()
