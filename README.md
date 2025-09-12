# machinemoo - Machine Learning Multi-Objective Optimization

`machinemoo` is a Python library designed for multi-objective optimization (MOO) with machine learning models. It provides modular components to facilitate scalarization strategies, optimization methods, ensembles, metrics, visualization, and utility functions — all aimed at simplifying research and application of MOO problems.

---

## Features

- **MOO Handler**: Implementations of popular MOO algorithms (NISE, MONISE, MOLA, Random Weights) with a unified interface.
- **Scalarization Module**: Base classes and examples to create custom scalarization methods that convert multi-objective problems into single-objective ones.
- **Analysis Module**: Metrics computation (e.g., hypervolume), ensemble learning integration, strategies for a posteriori decision-making, and visualization tools for Pareto frontiers and other MOO-related plots.
- **Utils**: Common utilities like centralized logging configuration to streamline development and debugging.

---

## Modules Overview

### 1. `moo`

Handles multi-objective optimization workflows.  
- Run different MOO algorithms via a single interface.  
- Retrieve optimization results, models, objectives, and hypervolume metrics.

### 2. `moo_scalarization`

Provides a base class to implement scalarization strategies.  
- Train models weighted by objectives.  
- Return objective values and optionally gradients for optimization.

### 3. `analysis`

Includes metrics, ensembles, and visualization tools.  
- Calculate hypervolume metrics.  
- Create ensembles from multiple models using different strategies.  
- Visualize Pareto frontiers (2D/3D/coordenate parallels for >= 4D), hypervolume evolution.

### 4. `utils`

Contains utility functions and configurations.  
- Centralized logging configuration to ensure consistent output and manage verbosity of external libraries.  
- Other general-purpose helpers to support the package.

---

## Installation

```bash
pip install .
```

or

```bash
poetry install
```

## Quick Start

```python
from machinemoo import moo
from machinemoo.scalarization import Scalarization
from machinemoo.analysis.visualization import plot_pareto

# Create a scalarization instance (custom implementation)
w_scalar = Scalarization(...)

# Run optimization using the MOLA method
moopt = moo(w_scalar).mo_optimization('mola', opt_params)

# Extract objectives
objs = moopt.get_objectives()

# Visualize Pareto frontier
plot_pareto({'mola': objs})
```

---

## Documentation

Detailed documentation is provided in each module:

- **`moo/`**  
  Implements optimization algorithms such as NISE, MONISE, MOLA, and Random Weights.  
  Offers a unified interface to execute and extract results from multi-objective optimization processes.

- **`moo_scalarization/`**  
  Contains the `Scalarization` base class to define how models are trained using weighted objectives.  
  Users can subclass it to create custom scalarization methods that return objective values and (optionally) gradients.

- **`analysis/`**  
  A toolbox for evaluating and interpreting results, including:
  - **`metrics.py`**: Hypervolume computation and performance tracking.
  - **`ensembles.py`**: Voting, Bagging, and Boosting ensembles for learned models.
  - **`visualization.py`**: 2D/3D/parallel coordinates plots for Pareto frontiers and metric evolution.

- **`utils/`**  
  Shared utilities for logging and global configuration:
  - `logging_config.py`: Sets up formatted console logging and suppresses verbosity from noisy libraries (e.g., sklearn, Pyomo).

---

## Contribution


---

## License