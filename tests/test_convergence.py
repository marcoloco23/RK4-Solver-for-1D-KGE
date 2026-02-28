"""Tests for the convergence analysis utilities."""

import numpy as np

from kg_solver.convergence import convergence_order, l2_error


class TestL2Error:
    def test_identical_arrays(self):
        a = np.ones(10)
        assert l2_error(a, a, dx=0.1) == 0.0

    def test_known_error(self):
        a = np.ones(5)
        b = np.zeros(5)
        dx = 1.0
        # sqrt(dx * sum(1^2 * 5)) = sqrt(5)
        expected = np.sqrt(5.0)
        assert abs(l2_error(a, b, dx) - expected) < 1e-12


class TestConvergenceOrder:
    def test_fourth_order(self):
        # If error halves with 2x refinement at 4th order: error_fine = error_coarse / 16
        order = convergence_order(16.0, 1.0, refinement_factor=2)
        assert abs(order - 4.0) < 1e-12

    def test_second_order(self):
        order = convergence_order(4.0, 1.0, refinement_factor=2)
        assert abs(order - 2.0) < 1e-12
