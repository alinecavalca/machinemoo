from typing import Optional, List, Dict
import numpy as np

from machinemoo.analysis.old.visualization import (
    plot_pareto_2d,
    plot_pareto_3d,
    plot_pareto_3d_multiple,
    plot_pareto_3d_multiples,
    plot_pareto_2d_multiple,
    plot_hypervolume,
    plot_multiple_hypervolumes,
    plot_parallel_coordinates,
    plot_parallel_coordinates_subplots, 
    plot_parallel_coordinates_all,
    plot_pareto_2d_set_limit
)
from machinemoo.utils.logging_config import logger

class Analyzer:
    """
    Class to help visualize and analyze the results from multi-objective optimization.
    """

    def run(
        objectives: np.ndarray,
        method: str = "",
        hypervolume_values: Optional[List[float]] = None,
        show_parallel: bool = True,
        show_pareto: bool = True,
        show_hypervolume: bool = True,
        axis_label: Optional[str] = None,
        point: Optional[np.ndarray] = None,
        color: str = None,
    ) -> None:
        """
        Run a standard visualization pipeline for MOO results.

        Parameters
        ----------
        objectives : np.ndarray
            Objective vectors (solutions) with shape (n_solutions, n_objectives).
        method : str
            Name of the method (used in plot titles/labels).
        hypervolume_values : list of float, optional
            Evolution of hypervolume (if available).
        show_parallel : bool
            Whether to plot the parallel coordinates chart.
        show_pareto : bool
            Whether to plot the Pareto frontier (2D/3D if applicable).
        show_hypervolume : bool
            Whether to plot the hypervolume evolution chart.
        point: bool
            Optional np.array with shape (2,) to highlight on the plot.
        """
        if objectives.shape[1] == 2 and show_pareto is True:
            if axis_label is None:
                plot_pareto_2d(
                                objectives, 
                                method=method, 
                                baseline=point,
                                color=color
                                )
            else:
                plot_pareto_2d(
                                objectives, 
                                method=method, 
                                baseline=point,
                                color=color,
                                x_label=axis_label[0],
                                y_label=axis_label[1]
                                )
        
        elif objectives.shape[1] == 3 and show_pareto is True:
            if axis_label is None:
                plot_pareto_3d(
                                objectives, 
                                method=method,
                                point=point,
                                color=color
                                )
            else:
                plot_pareto_3d(
                                objectives, 
                                method=method,
                                point=point,
                                color=color,
                                x_label=axis_label[0],
                                y_label=axis_label[1]
                                )
                

        if show_hypervolume and hypervolume_values is not None:
            plot_hypervolume(hypervolume_values, method=method, color=color)
 
        if show_parallel:
            plot_parallel_coordinates(objectives, method=method)

        print("[Analyzer] Visualization complete.")

    def run_multiple(
        pareto_dict: Dict[str, np.ndarray] =None,
        hypervolumes_values: Optional[Dict[str, List[float]]] = None,
        show_pareto: bool = False,
        show_hypervolume: bool = True,
        show_parallel: bool = False,
        subplots: bool = False,
        subplots_hv: bool = False,
        axis_label: Optional[str] = None,
        point: Optional[np.ndarray] = None,
        colors_pareto: Optional[np.ndarray] = None,
        colors_hv: Optional[np.ndarray] = None,
        num_views: int = 1,
    ) -> None:
        """
        Run a standard visualization pipeline for MOO results.

        Parameters
        ----------
        objectives : np.ndarray
            Objective vectors (solutions) with shape (n_solutions, n_objectives).
        method : str
            Name of the method (used in plot titles/labels).
        hypervolume_values : list of float, optional
            Evolution of hypervolume (if available).
        show_parallel : bool
            Whether to plot the parallel coordinates chart.
        show_pareto : bool
            Whether to plot the Pareto frontier (2D/3D if applicable).
        show_hypervolume : bool
            Whether to plot the hypervolume evolution chart.
        point: bool
            Optional np.array with shape (2,) to highlight on the plot.
        """
        if not pareto_dict:
            raise ValueError("pareto_dict is empty.")

         # Detect number of objectives (assume all methods have the same)
        first_method = next(iter(pareto_dict))
        objectives = pareto_dict[first_method]
        num_objectives = objectives.shape[1]

        # Check if all methods have same number of objectives
        for method, obj_array in pareto_dict.items():
            if obj_array.shape[1] != num_objectives:
                raise ValueError(f"Method '{method}' has {obj_array.shape[1]} objectives, expected {num_objectives}.")

        # Warn if number of solutions differs
        num_solutions_set = {obj_array.shape[0] for obj_array in pareto_dict.values()}
        if len(num_solutions_set) > 1:
            logger.warning(f"[Methods have different numbers of solutions: {num_solutions_set}")

        if num_objectives == 2 and show_pareto is True:
            if axis_label is None:
                plot_pareto_2d_set_limit(
                                        pareto_dict=pareto_dict, 
                                        show_subplot=subplots, 
                                        point=point,
                                        colors=colors_pareto
                                        )
            else:
                plot_pareto_2d_set_limit(
                                        pareto_dict=pareto_dict, 
                                        show_subplot=subplots, 
                                        point=point,
                                        colors=colors_pareto,
                                        x_label=axis_label[0],
                                        y_label=axis_label[1]
                                        )    
        elif num_objectives == 3 and show_pareto is True:
            if axis_label is None:
                plot_pareto_3d_multiples(
                                        pareto_dict=pareto_dict,
                                        point=point,
                                        colors=colors_pareto
                                        )
            else:
                plot_pareto_3d_multiples(
                                        pareto_dict=pareto_dict,
                                        point=point,
                                        colors=colors_pareto,
                                        x_label=axis_label[0],
                                        y_label=axis_label[1],
                                        z_label=axis_label[2]
                                        )
        # elif show_pareto and num_objectives not in [2, 3]: 
        #     logger.info("Pareto plot not supported for more than 3 objectives. Try plot parallel coordinates")

        if show_hypervolume and hypervolumes_values is not None:
            plot_multiple_hypervolumes(hypervolumes_values, subplots_hv, colors=colors_hv)
            plot_multiple_hypervolumes(hypervolumes_values, False, colors=colors_hv)

        if show_parallel:
            plot_parallel_coordinates_subplots(pareto_dict)
            plot_parallel_coordinates_all(pareto_dict, colors=colors_pareto)

        print("[Analyzer] Visualization complete.")