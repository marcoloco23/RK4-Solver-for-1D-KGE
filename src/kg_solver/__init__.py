"""Klein-Gordon equation solver using RK4 time integration and finite differences."""

from kg_solver.domain import Domain
from kg_solver.solver import solve

__all__ = ["Domain", "solve"]
__version__ = "1.0.0"
