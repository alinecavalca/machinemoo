# Analysis Module

The `analysis` module provides essential tools for a posteriori analysis and evaluating multi-objective optimization results. It includes:

- **Metrics calculation** — such as hypervolume for quality assessment.
    - Currently supports Hypervolume metric
- **Ensemble methods** — combining multiple models for improved predictions.
    - Currently supports Scikit-Learn based ensembles: Voting, Bagging, Adaboost
- **Visualization tools** — comprehensive plotting functions to analyze Pareto frontiers and hypervolume evolution.

---

## Submodules

### 1. `metrics.py`

Contains functions to calculate key metrics for multi-objective optimization, currently including:

- **Hypervolume calculation**: Measures the volume covered by the Pareto frontier, helping evaluate the quality of solutions.

**Example:**

```python
from machinemoo.analysis.metrics import calculate_hypervolume

hypervolume = calculate_hypervolume(solutions, reference_point)
print(f"Hypervolume: {hypervolume}")
```

---

### 2. `ensembles.py`

Implements ensemble learning methods to combine multiple models. Supported ensemble types include:

- Voting classifiers (hard and soft voting)
- Bagging classifiers
- AdaBoost classifiers

**Example:**

```python
from machinemoo.analysis.ensembles import Ensemble

ensemble = Ensemble(
    models=model_list,
    X_train=X_train,
    y_train=y_train,
    ensemble_type='voting',
    voting_type='soft'
)
predictions = ensemble.predict(X_test)
```

---

### 3. `visualization.py`

Provides a comprehensive set of plotting functions to analyze multi-objective optimization results. Features include:

- Pareto frontier visualization in 2D, 3D, and parallel coordinates (for >= 4D).
- Hypervolume evolution plotting.
- Margin (mu) evolution plotting.
- Support for Matplotlib and Plotly backends (for pareto frontier plots).

**Example:**

```python
from machinemoo.analysis.visualization import plot_pareto, plot_hypervolume, plot_mu_evolution

# Plot Pareto frontiers for multiple methods
plot_pareto(methods_solutions, labels=["Objective 1", "Objective 2"])

# Plot hypervolume evolution
plot_hypervolume(hv_values, method="MOLA")

# Plot margin (mu) evolution
plot_mu_evolution(mu_values, title="Margin Evolution")
