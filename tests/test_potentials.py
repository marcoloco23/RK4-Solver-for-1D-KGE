"""Tests for potential functions and their derivatives."""

import numpy as np
import pytest

from kg_solver.potentials import (
    free_field_derivative,
    free_field_potential,
    phi4_derivative,
    phi4_potential,
    sine_gordon_derivative,
    sine_gordon_potential,
)

PHI = np.linspace(-2, 2, 200)
EPS = 1e-6


def _numerical_derivative(V, phi, eps=EPS):
    """Central finite-difference approximation of dV/dphi."""
    return (V(phi + eps) - V(phi - eps)) / (2 * eps)


class TestPhi4:
    def test_derivative_matches_potential(self):
        dV_numerical = _numerical_derivative(phi4_potential, PHI)
        dV_analytic = phi4_derivative(PHI)
        np.testing.assert_allclose(dV_analytic, dV_numerical, atol=1e-5)

    def test_minima_at_plus_minus_one(self):
        assert phi4_potential(np.array([1.0]))[0] == pytest.approx(0.0)
        assert phi4_potential(np.array([-1.0]))[0] == pytest.approx(0.0)

    def test_derivative_zero_at_minima(self):
        assert phi4_derivative(np.array([1.0]))[0] == pytest.approx(0.0, abs=1e-15)
        assert phi4_derivative(np.array([-1.0]))[0] == pytest.approx(0.0, abs=1e-15)


class TestSineGordon:
    def test_derivative_matches_potential(self):
        dV_numerical = _numerical_derivative(sine_gordon_potential, PHI)
        dV_analytic = sine_gordon_derivative(PHI)
        np.testing.assert_allclose(dV_analytic, dV_numerical, atol=1e-5)

    def test_minimum_at_zero(self):
        assert sine_gordon_potential(np.array([0.0]))[0] == pytest.approx(0.0)


class TestFreeField:
    def test_derivative_matches_potential(self):
        dV_numerical = _numerical_derivative(free_field_potential, PHI)
        dV_analytic = free_field_derivative(PHI)
        np.testing.assert_allclose(dV_analytic, dV_numerical, atol=1e-5)

    def test_custom_mass(self):
        m = 2.5
        V = lambda phi: free_field_potential(phi, m=m)
        dV_numerical = _numerical_derivative(V, PHI)
        dV_analytic = free_field_derivative(PHI, m=m)
        np.testing.assert_allclose(dV_analytic, dV_numerical, atol=1e-5)
