import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from machinemoo import moo
from machinemoo import MooScalarization
from machinemoo.analysis.metrics import compute_hypervolume_progress
from machinemoo.analysis.visualization import plot_pareto_2d, plot_parallel_coordinates, plot_hypervolume
from machinemoo.analysis.moo_analyzer import Analyzer

# --- Load and split dataset ---
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Define scalarization ---
weighted_scalar = MooScalarization(base_model=RandomForestClassifier, X=X_train, y=y_train)
single_scalar = None  # optional, defaults to weighted_scalar

# --- Run multi-objective optimization ---
optimizer = moo(weighted_scalar=weighted_scalar, single_scalar=single_scalar)
solver = optimizer.mo_optimization(method="ml_moo", params={"n_iter": 20})

# --- Extract objective values ---
objectives = optimizer.get_objectives(solver)

# --- Basic plots ---
plot_pareto_2d(objectives)
plot_parallel_coordinates(objectives)
compute_hypervolume_progress(objectives, reference_point=np.ones(objectives.shape[1]))

# --- Run ensemble prediction ---
accuracy = optimizer.run_ensemble(
    solver,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    ensemble_type="voting",
    voting_type="soft"
)

# --- All-in-one pipeline analysis ---
#analyze_results(objectives, method="ML-MOO")
