"""Tests for the RK4 solver core."""

import numpy as np
import pytest

from kg_solver.domain import Domain
from kg_solver.initial_conditions import boosted_kink, static_kink
from kg_solver.potentials import phi4_derivative
from kg_solver.solver import solve


class TestStaticKink:
    """A static kink is an exact stationary solution of the phi^4 KGE.

    It should remain nearly unchanged under time evolution (up to
    discretisation error at the boundaries).
    """

    def test_kink_stays_stationary(self):
        domain = Domain(x_min=-10, x_max=10, nx=201, t_max=2.0, nt=400)
        phi_0, dphi_dt_0 = static_kink(domain)

        phi, _ = solve(domain, phi_0, dphi_dt_0, phi4_derivative)

        # Compare final state to initial (excluding boundary effects)
        interior = slice(10, -10)
        error = np.max(np.abs(phi[-1, interior] - phi_0[interior]))
        assert error < 1e-3, f"Static kink drifted: max error = {error:.2e}"


class TestBoostedKink:
    """A Lorentz-boosted kink has a known exact solution."""

    def test_boosted_kink_accuracy(self):
        domain = Domain(x_min=-15, x_max=15, nx=301, t_max=2.0, nt=800)
        beta = 0.3
        phi_0, dphi_dt_0 = boosted_kink(domain, beta=beta)

        phi, _ = solve(domain, phi_0, dphi_dt_0, phi4_derivative)

        # Exact solution at final time
        x = domain.x
        t_final = domain.t_max
        gamma = 1.0 / np.sqrt(1 - beta**2)
        xi = np.sqrt(2) * gamma * (x - beta * t_final)
        phi_exact_final = np.tanh(xi)

        interior = slice(20, -20)
        error = np.max(np.abs(phi[-1, interior] - phi_exact_final[interior]))
        assert error < 5e-3, f"Boosted kink error = {error:.2e}"


class TestConvergenceOrder:
    """Grid refinement convergence using the boosted kink exact solution."""

    def test_spatial_convergence(self):
        """Verify that halving dx reduces error, consistent with high-order scheme."""
        beta = 0.3
        errors = []
        nx_values = [101, 201]

        for nx in nx_values:
            nt = 10 * nx  # keep CFL roughly constant
            domain = Domain(x_min=-15, x_max=15, nx=nx, t_max=1.0, nt=nt)
            phi_0, dphi_dt_0 = boosted_kink(domain, beta=beta)
            phi_num, _ = solve(domain, phi_0, dphi_dt_0, phi4_derivative)

            # Exact solution at final time
            x = domain.x
            gamma = 1.0 / np.sqrt(1 - beta**2)
            xi = np.sqrt(2) * gamma * (x - beta * domain.t_max)
            phi_exact_final = np.tanh(xi)

            # L2 error on interior (avoid boundary artifacts)
            interior = slice(20, -20)
            diff = phi_num[-1, interior] - phi_exact_final[interior]
            err = np.sqrt(domain.dx * np.sum(diff**2))
            errors.append(err)

        order = np.log2(errors[0] / errors[1])
        assert order > 1.5, f"Convergence order {order:.2f} is too low"


class TestSolverOutputShape:
    """Basic shape and boundary checks."""

    def test_output_shape(self):
        domain = Domain(x_min=0, x_max=1, nx=11, t_max=0.1, nt=10)
        phi_0 = np.sin(np.pi * domain.x)
        dphi_dt_0 = np.zeros(domain.nx)

        phi, pi = solve(domain, phi_0, dphi_dt_0, phi4_derivative)

        assert phi.shape == (domain.nt + 1, domain.nx)
        assert pi.shape == (domain.nt + 1, domain.nx)

    def test_initial_condition_preserved(self):
        domain = Domain(x_min=0, x_max=1, nx=11, t_max=0.1, nt=10)
        phi_0 = np.sin(np.pi * domain.x)
        dphi_dt_0 = np.zeros(domain.nx)

        phi, pi = solve(domain, phi_0, dphi_dt_0, phi4_derivative)

        np.testing.assert_array_equal(phi[0], phi_0)
        np.testing.assert_array_equal(pi[0], dphi_dt_0)
