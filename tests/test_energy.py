"""Tests for energy computation and conservation."""

import numpy as np

from kg_solver.domain import Domain
from kg_solver.energy import energy_timeseries, total_energy
from kg_solver.initial_conditions import static_kink
from kg_solver.potentials import phi4_derivative, phi4_potential
from kg_solver.solver import solve


class TestEnergyConservation:
    """Energy should be conserved to within O(dt^4) for RK4."""

    def test_static_kink_energy_conserved(self):
        domain = Domain(x_min=-10, x_max=10, nx=201, t_max=2.0, nt=400)
        phi_0, dphi_dt_0 = static_kink(domain)

        phi, pi = solve(domain, phi_0, dphi_dt_0, phi4_derivative)
        E = energy_timeseries(phi, pi, domain, phi4_potential)

        # Energy should stay nearly constant
        relative_drift = np.abs(E[-1] - E[0]) / E[0]
        assert relative_drift < 1e-4, f"Energy drift: {relative_drift:.2e}"

    def test_energy_positive(self):
        domain = Domain(x_min=-10, x_max=10, nx=201, t_max=1.0, nt=200)
        phi_0, dphi_dt_0 = static_kink(domain)

        E = total_energy(phi_0, dphi_dt_0, domain, phi4_potential)
        assert E > 0, f"Energy should be positive for kink, got {E}"
