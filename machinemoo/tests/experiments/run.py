from typing import Any, Dict, List, Tuple

import numpy as np

from machinemoo import MooScalarization, get_objectives, moo
from machinemoo.analysis.metrics import compute_hypervolume_progress

__all__ = ["run_experiments"]


def run_experiments(
    methods: List[str],
    opt_params: Dict[str, Any],
    model: Any = None,
    train_fn: Any = None,
    num_objs: int = 2,
    moo_instance: moo = None,
    store_models: bool = True,
    mola_hv: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Run multiple MOO methods and collect results for analysis.

    Parameters
    ----------
    methods : list of str
        Names of optimization methods to run.
    opt_params : dict
        Parameters to pass to each optimization method.
    moo_instance : MachineMoo
        Initialized MachineMoo object with scalarizations set.
    store_models : bool
        Whether to store the models trained during optimization.

    Returns
    -------
    dict
        Dictionary containing results per method.
    """
    results = {}
    aux = None
    if moo_instance is not None:
        aux = moo_instance

    for method in methods:
        print(f"Running {method.upper()}...")
        w_scalar = 0

        if moo_instance is None:
            w_scalar = MooScalarization(model, train_fn, num_objs)
            moo_instance = moo(w_scalar)
        try:
            solver = moo_instance.mo_optimization(method=method, params=opt_params)
            objectives_solutions = get_objectives(solver)
            hypervolume_values = compute_hypervolume_progress(objectives_solutions)

            if mola_hv is True:
                hypervolume_values = solver.get_hypervolumes()

            results[method] = {
                "solver": solver,
                "objectives": objectives_solutions,
                "models": (
                    solver.get_models()
                    if store_models and hasattr(solver, "get_models")
                    else None
                ),
                "hypervolume": hypervolume_values,
            }

            moo_instance = aux

        except Exception as e:
            print(f"Failed to run {method}: {e}")
            results[method] = {"error": str(e)}

    pareto_dict, hv_dict = extract_pareto_and_hv_dicts(results)

    return results, pareto_dict, hv_dict


def extract_pareto_and_hv_dicts(
    experiment_results: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[float]]]:
    """
    Extract dictionaries for Pareto front objectives and hypervolume values
    from experiment results.

    Args:
        experiment_results: Dictionary where each key is a method name,
                            and the value is another dict with keys like
                            "objectives" and "hypervolume".

    Returns:
        A tuple containing:
        - pareto_dict: method -> npt.NDArray[np.float64] of objective values
        - hypervolumes_dict: method -> list of hypervolume values
    """
    pareto_dict = {
        method: res["objectives"]
        for method, res in experiment_results.items()
        if "objectives" in res
    }

    hypervolumes_dict = {
        method: res["hypervolume"]
        for method, res in experiment_results.items()
        if "hypervolume" in res
    }

    return pareto_dict, hypervolumes_dict
