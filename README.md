# Klein-Gordon Equation Solver

A modular RK4 solver for the 1+1D Klein-Gordon equation with pluggable potentials, exact-solution validation, and built-in convergence analysis.

## The equation

The Klein-Gordon equation in one spatial dimension:

```
d²phi/dt² = d²phi/dx² - dV/dphi
```

where `V(phi)` is a potential function. This solver supports:

| Potential | V(phi) | dV/dphi |
|-----------|--------|---------|
| phi^4 double-well | (phi² - 1)² | 4 phi (phi² - 1) |
| Sine-Gordon | 1 - cos(phi) | sin(phi) |
| Free massive field | m²phi²/2 | m²phi |

Custom potentials are supported — just pass any callable `dV(phi) -> array`.

## Method

- **Spatial discretisation**: Second-order central finite differences
- **Time integration**: Classical 4th-order Runge-Kutta (method of lines)
- **Boundary conditions**: Fixed Dirichlet (configurable per initial condition)

## Installation

```bash
pip install -e .
```

For development (includes pytest and ruff):

```bash
pip install -e ".[dev]"
```

Requires Python >= 3.10, NumPy >= 1.24, Matplotlib >= 3.7.

## Quick start

```python
import numpy as np
from kg_solver import Domain, solve
from kg_solver.potentials import phi4_derivative
from kg_solver.initial_conditions import static_kink

# Define the grid
domain = Domain(x_min=-10, x_max=10, nx=201, t_max=5.0, nt=1000)

# Set initial conditions and solve
phi_0, dphi_dt_0 = static_kink(domain)
phi, pi = solve(domain, phi_0, dphi_dt_0, phi4_derivative)

# phi[t_index, x_index] contains the full spacetime solution
```

## Available initial conditions

```python
from kg_solver.initial_conditions import (
    gaussian,                  # Gaussian pulse at rest
    sine_mode,                 # Standing sine mode
    vacuum,                    # Uniform vacuum (+/-1)
    static_kink,               # Stationary phi^4 kink
    boosted_kink,              # Lorentz-boosted kink
    kink_antikink_collision,   # Two solitons colliding
)
```

## Diagnostics

**Energy conservation**:

```python
from kg_solver.energy import energy_timeseries
from kg_solver.potentials import phi4_potential

E = energy_timeseries(phi, pi, domain, phi4_potential)
# E[n] = total energy at time step n — should be nearly constant
```

**Grid convergence study**:

```python
from kg_solver.convergence import convergence_study

results = convergence_study(base_domain, ic_factory, dV, exact_solution)
print(results["orders"])  # Should approach 4.0 for RK4 + 2nd-order FD
```

## Visualization

```python
from kg_solver.visualization import plot_spacetime, animate_solution

plot_spacetime(phi, domain)           # Pseudocolour x-t plot
anim = animate_solution(phi, domain)  # Matplotlib animation
```

## Examples

Full runnable scripts in `examples/`:

```bash
python examples/gaussian_pulse.py      # Dispersive wave evolution
python examples/kink_collision.py       # Soliton scattering + convergence
```

## Running tests

```bash
pytest -v
```

18 tests covering: potential derivatives, solver accuracy (static kink, boosted kink, convergence order), energy conservation, and utility functions.

## Project structure

```
src/kg_solver/
    domain.py              Grid configuration (immutable dataclass)
    potentials.py          Pluggable V(phi) and dV/dphi
    initial_conditions.py  IC factories returning (phi_0, dphi_dt_0)
    solver.py              Vectorized RK4 integrator
    energy.py              Energy density and conservation
    convergence.py         L2 error and grid refinement analysis
    visualization.py       Plotting and animation utilities
tests/                     pytest suite
examples/                  Runnable demo scripts
```

## License

MIT
