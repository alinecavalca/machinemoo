# Multi-Objective Optimization (MOO) Module

This module provides a unified interface for solving multi-objective optimization problems using scalarization-based methods.

## ✅ Supported Methods

- `NISE`: Non-Inferior Set Estimation (NISE)
    - A classical scalarization method that iteratively finds Pareto optimal solutions by solving weighted sum problems with carefully chosen weight vectors. Efficient for two-objective problems and provides a convex approximation of the Pareto front.
- `MONISE`: Many-Objective NISE extension
    - An extension of NISE designed to handle many-objective problems (more than two objectives). It generalizes the weight selection process to explore the Pareto front in higher dimensions while maintaining computational efficiency.
- `MOLA`: Multi-objective Learning Algorithm
    - A machine learning based method that approximates the Pareto front by training scalarization models. It adaptively updates weights and solutions to improve the front coverage, suitable for complex or high-dimensional problems, as non-convex problems (e.g. deep learning).
- `Random Weight`: Randomly generated weights to explore the Pareto front
    - A straightforward baseline method that samples random weight vectors to scalarize and optimize objectives. Though simple, it can provide diverse Pareto solutions but may be less efficient or precise than structured methods.

## 📦 Installation

Make sure your environment is configured and `machinemoo` is installed and properly set up. Then, simply import and use:

```bash
from machinemoo.moo.moo_handler import moo
```

## 🚀 Quick Start

You can optimize using any supported method by specifying its name and parameters:

```python
# Define your scalarization object
w_scalar = YourWeightedScalarization()

# Define optimization parameters
opt_params = {
    'node_time_limit': 2,
    'target_size': 300,
    'target_gap': 0,
    'node_gap': 0.05,
    'norm': False
}

# Run optimization
moopt = moo(w_scalar).mo_optimization(method="mola", **opt_params)
```

Alternatively, you can call a specific method directly:

```python
moopt = moo(w_scalar).mola(**opt_params)
```

### 🔎 Logging options

When creating a `moo` instance, you can control the logging behavior using the parameters `verbose` and `debug`:

```python
moopt = moo(w_scalar, verbose=True, debug=False).mo_optimization(method, **opt_params)
```

- `verbose=True` → enables **INFO** messages (progress and general information).  
- `debug=True` → enables **DEBUG** messages (detailed internal steps). This takes priority over `verbose`.  
- If both are set to `False`, only **ERROR** messages will be shown (silent mode).  


## 🔧 Parameters

Each method supports slightly different parameters, but most share:

| Parameter         | Type    | Description                                      |
|------------------|---------|--------------------------------------------------|
| `node_time_limit`| `float` | Maximum time allowed per scalarization node (in seconds). |
| `target_size`    | `int`   | Target number of solutions to collect from the optimization process. |
| `target_gap`     | `float` | Minimum improvement threshold to continue exploring new solutions. |
| `node_gap`       | `float` | Scalarization-specific tolerance gap (e.g., in MONISE). |
| `norm`           | `bool`  | Whether to normalize the objective vectors during the optimization. |

Additional parameters might be supported depending on the method implementation.

## 📊 Accessing Results

After running optimization methods via `moo_handler`, you can easily retrieve results and metadata from the `moopt` object.

Example:

```python
objs = get_objectives(moopt)
hypervolume_values = moopt.get_hypervolumes()

results[method] = {
    "moopt": moopt,
    "objectives": objs,
    "models": get_models(moopt),
    "hypervolume": hypervolume_values
}
```

Here:

- `get_objectives(moopt)` retrieves the list of objective vectors found.
- `moopt.get_hypervolumes()` returns the hypervolume progression during optimization.
- `get_models(moopt)` obtains the trained models corresponding to each solution.

This structure facilitates easy comparison and analysis of different methods’ results.

## 📎 Notes

- The object `moopt` follows a consistent interface across methods.
- Scalarization weights and decision vectors are also stored internally.
- Visualization and analysis tools are available in the `analysis` module.

## Usage Tips

* Random Weights method is slow for many objectives but finds the optimal solution.
* MONISE method is better in some cases but only is limit to handle with convex problem.