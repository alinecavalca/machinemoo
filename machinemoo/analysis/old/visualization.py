# ml_moo/analysis/visualization.py

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy.typing as npt

from typing import Optional, List, Tuple, Sequence, Union, Dict, Any
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import parallel_coordinates

plt.style.use("seaborn-v0_8-whitegrid")

def plot_pareto_2d(
    objectives: List[List[float]],
    labels: Optional[List[str]] = None,
    title: str = "Pareto Frontier (2D)",
    point: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    color: str = "#FF5C8D"
) -> None:
    """
    Plots a 2D Pareto frontier.

    Args:
        objectives: List of [obj1, obj2] points.
        labels: Optional labels for x and y axis.
        title: Plot title.
        point: Optional np.array with shape (2,) to highlight on the plot.
        save_path: Optional path to save the figure.
        color: Color palette for scatter.
    """
    df = pareto_to_dataframe(objectives, labels)
    x_label, y_label = labels if labels else ["Objective 1", "Objective 2"]

    plt.figure(figsize=(6, 5))
    #palette = sns.color_palette(color, n_colors=10)
    #plt.scatter(df["obj1"], df["obj2"], color=palette[-2], label="Pareto Solutions")
    plt.scatter(df["obj1"], df["obj2"], color=color, edgecolors='black', linewidths=0.5, label="Pareto Solutions")

    if point is not None:
        point = np.asarray(point).flatten()
        plt.scatter([point[0]], [point[1]], color="#FFC145", edgecolors='black', linewidths=0.5, label="Baseline")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")

    plt.show()

def plot_pareto_2d_cinza(
    objectives: List[List[float]],
    labels: Optional[List[str]] = None,
    title: str = "Pareto Frontier (2D)",
    point: Optional[np.ndarray] = None,
    point_label: Optional[List[str]] = "Baseline",
    save_path: Optional[str] = None,
    color: str = "gray",
    show: bool = True,
) -> None:
    """
    Plots a 2D Pareto frontier.

    Args:
        objectives: List of [obj1, obj2] points.
        labels: Optional labels for x and y axis.
        title: Plot title.
        point: Optional point to highlight (e.g., baseline solution).
        point_label: Optional label for point baseline.
        save_path: Optional path to save the figure.
        color: Color palette for scatter.
    """
    if objectives.shape[1] != 2:
        raise ValueError("Expected 2D input for plot_pareto_2d.")

    x_label, y_label = labels if labels else ["Objective 1", "Objective 2"]

    plt.figure(figsize=(6, 4))
    plt.scatter(objectives[:, 0], objectives[:, 1], c="steelblue", s=60, edgecolors="black", alpha=1.0)
    
    if point is not None:
        point = np.asarray(point).flatten()
        plt.scatter([point[0]], [point[1]], color="indianred", s=80, label=point_label)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()
    plt.close()

def plot_pareto_3d(
    objectives: np.ndarray,
    labels: Optional[List[str]] = ["Objective 1", "Objective 2", "Objective 3"],
    title: str = "Pareto Front (3D)",
    save_path: Optional[str] = None,
    show: bool = True,
    point: Optional[np.ndarray] = None,
    color: str = "gray"
) -> None:
    """
    Plots a 3D Pareto front.

    Args:
        objectives (np.ndarray): Array with shape (n_points, 3) representing objective values.
        labels (Tuple[str, str, str], optional): Axis labels for the objectives.
        title (str): Title of the plot.
        save_path (str, optional): Path to save the figure. If None, doesn't save.
        show (bool): Whether to display the plot.

    Returns:
        None
    """
    if objectives.shape[1] != 3:
        raise ValueError("Expected 3D input for plot_pareto_3d.")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    if point is not None:
        point = np.asarray(point).flatten()
        ax.scatter([point[0]], [point[1]], [point[2]], color="#FFC145", edgecolors='black', s=30, linewidths=0.5, label="Baseline")
    
    ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2], color=color, s=30, edgecolor="black", linewidths=0.5)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.set_title(title)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()
    plt.close()

def plot_pareto_3d_plotly_test(
    objectives: Sequence[Sequence[float]],
    labels: Optional[List[str]] = None,
    title: str = "Pareto Frontier (3D Plotly)",
    baseline: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Plots a 3D Pareto frontier using Plotly.

    Args:
        objectives: List of 3D objective vectors.
        labels: Axis labels.
        title: Plot title.
        baseline: Optional baseline point.

    Returns:
        DataFrame with objective values.
    """
    df = pd.DataFrame(objectives, columns=labels or [f"Objective {i+1}" for i in range(3)])
    x, y, z = df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(size=5, color=z, colorscale="Viridis", opacity=0.8, line=dict(width=1, color="black")),
        name="Pareto Frontier"
    ))

    if baseline is not None and len(baseline) == 3:
        fig.add_trace(go.Scatter3d(
            x=[baseline[0]], y=[baseline[1]], z=[baseline[2]],
            mode="markers",
            marker=dict(size=8, color="red"),
            name="Baseline"
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=df.columns[0],
            yaxis_title=df.columns[1],
            zaxis_title=df.columns[2]
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()
    return df

def pareto_to_dataframe(
    objectives: List[List[np.ndarray]],
    labels: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Converts a list of objectives into a sorted DataFrame.

    Args:
        objectives: List of [obj1, obj2] values.
        labels: Optional column names.

    Returns:
        Sorted DataFrame by first objective.
    """
    default_labels = ["obj1", "obj2"]
    cols = labels if labels else default_labels
    df = pd.DataFrame(objectives, columns=default_labels)
    return df.sort_values(by=default_labels[0])

def plot_parallel_coordinates(
    objectives,
    labels: Optional[List[str]] = None,
    title: str = "Parallel Coordinates Plot",
    colormap: str = "viridis",
    color: str = None,
    alpha: float = 1.0,
    show_legend: bool = False,
    best_solution: Optional[npt.NDArray[np.float64]] = None,
    best_sol_label: str = "Chosen solution"
) -> None:
    """
    Plots a colored parallel coordinates chart for multi-objective solutions.

    Args:
        objectives: A list of objective values for each solution.
        labels: Optional list of names for each objective.
        title: Title of the plot.
        colormap: Matplotlib colormap name.
        alpha: Transparency of lines (0 to 1).
        show_legend: Whether to display the legend (default False).

    Returns:
        None
    """
    num_objectives = len(objectives[0])
    default_labels = [f"Objective {i+1}" for i in range(num_objectives)]
    columns = labels if labels else default_labels

    df = pd.DataFrame(objectives, columns=columns)
    df["Solution"] = [f"S{i+1}" for i in range(len(df))]

    if color is None:
        color = sns.color_palette()
    plt.figure(figsize=(12, 6))
    parallel_coordinates(
        df,
        class_column="Solution",
        color=color,
        alpha=alpha
    )
    if best_solution is not None:
        best_solution = np.array(best_solution).reshape(1, -1)
        df_choosen = pd.DataFrame(best_solution, columns=columns)
        df_choosen["Solution"] = [best_sol_label]
        parallel_coordinates(
            df_choosen,
            class_column="Solution",
            color="green",
            alpha=alpha
        )

    plt.title(title, fontsize=16)
    plt.xlabel("Objectives", fontsize=14)
    plt.ylabel("Values", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(rotation=0)

    if not show_legend:
        plt.legend([], [], frameon=False)

    plt.tight_layout()
    plt.show()

def plot_parallel_coordinates_colored(
    objectives,
    labels: Optional[List[str]] = None,
    title: str = "Parallel Coordinates Plot",
    colormap: str = "viridis",
    alpha: float = 1.0,
    show_legend: bool = False,
    baseline: Optional[Union[List[float], np.ndarray]] = None,
    baseline_color: str = "black",
    baseline_linewidth: float = 3.0,
) -> None:
    """
    Plots a colored parallel coordinates chart for multi-objective solutions,
    with an optional highlighted baseline.

    Args:
        objectives: A list of objective values for each solution.
        labels: Optional list of names for each objective.
        title: Title of the plot.
        colormap: Matplotlib colormap name.
        alpha: Transparency of lines (0 to 1).
        show_legend: Whether to display the legend.
        baseline: Optional baseline objective to highlight.
        baseline_color: Color for the baseline line.
        baseline_linewidth: Thickness of the baseline line.

    Returns:
        None
    """
    num_objectives = len(objectives[0])
    default_labels = [f"Objective {i+1}" for i in range(num_objectives)]
    columns = labels if labels else default_labels

    df = pd.DataFrame(objectives, columns=columns)
    df["Solution"] = [f"S{i+1}" for i in range(len(df))]

    plt.figure(figsize=(12, 6))
    parallel_coordinates(
        df,
        class_column="Solution",
        colormap=plt.get_cmap(colormap),
        alpha=alpha,
        linewidth=1.5
    )

    if baseline is not None:
        if isinstance(baseline, np.ndarray):
            baseline = baseline.tolist()
        baseline_df = pd.DataFrame([baseline], columns=columns)
        baseline_df["Solution"] = ["Baseline"]
        parallel_coordinates(
            baseline_df,
            class_column="Solution",
            color=[baseline_color],
            linewidth=baseline_linewidth,
            alpha=1.0
        )

    plt.title(title, fontsize=16)
    plt.xlabel("Objectives", fontsize=14)
    plt.ylabel("Values", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(rotation=0)

    if not show_legend:
        plt.legend([], [], frameon=False)

    plt.tight_layout()
    plt.show()

def plot_parallel_coordinates_subplots(pareto_dict: dict[str, Any],
    method_order: list[str] = None,
    normalize: bool = True,
    figsize: tuple = (12, 4)
):
    """
    Plot multiple parallel coordinate plots in subplots for different methods.

    Parameters
    ----------
    pareto_dict : dict[str, list[list[float]]]
        Dictionary mapping method name to list of objective vectors.
    method_order : list[str], optional
        Order in which to plot the methods.
    normalize : bool, optional
        Whether to normalize objective values to [0, 1] per axis.
    figsize : tuple
        Figure size of the entire plot.
    """
    method_order = method_order or list(pareto_dict.keys())
    n_methods = len(method_order)

    fig, axes = plt.subplots(1, n_methods, figsize=figsize, sharey=False)

    if n_methods == 1:
        axes = [axes]

    for ax, method in zip(axes, method_order):
        objs = pareto_dict[method]
        df = pd.DataFrame(objs, columns=[f"Obj{i+1}" for i in range(len(objs[0]))])
        df["Method"] = method

        if normalize:
            df.iloc[:, :-1] = (df.iloc[:, :-1] - df.iloc[:, :-1].min()) / (df.iloc[:, :-1].max() - df.iloc[:, :-1].min())

        parallel_coordinates(df, class_column="Method", color=['tab:blue'], ax=ax)
        ax.set_title(method)
        ax.set_xticklabels(df.columns[:-1], rotation=45)

    plt.tight_layout()
    plt.show()

def plot_mu_evolution(mu_values: Sequence[float], title: str = "Margin Evolution") -> None:
    """
    Plots the evolution of margin (mu) across iterations.

    Args:
        mu_values: List of mu values.
        title: Plot title.
    """
    plt.figure(figsize=(6, 4))
    plt.plot(mu_values, color="purple", linewidth=2)
    plt.xlabel("Iterations")
    plt.ylabel("mu")
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_hypervolume(
    hypervolume_values: List[float],
    method: str = "",
    color: str = "tab:pink"
) -> None:
    """
    Plot the evolution of hypervolume for a single method.

    Args:
        hypervolume_values: List of hypervolume values over iterations.
        method: Label for the method (used in legend and title).
        color: Matplotlib-compatible color string.
    """
    plt.figure(figsize=(6, 5))
    plt.plot(hypervolume_values, label=f"Method {method}", color=color, linewidth=2)
    plt.xlabel("Iterations")
    plt.ylabel("Hypervolume")
    plt.title(f"Hypervolume Evolution - {method}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_multiple_hypervolumes(
    methods_hv: Dict[str, List[float]],
    subplots: bool = False,
    colors: Optional[str] = None,
) -> None:
    """
    Plot the evolution of hypervolume for multiple methods.

    Args:
        methods_hv: Dictionary where keys are method names and
                    values are lists of hypervolume values.
        subplots: Whether to plot each method in a separate subplot.
    """
    if colors is None:
        colors = plt.cm.tab10.colors
    #method_names = list(methods_hv.keys())

    if subplots:
        n_methods = len(methods_hv)
        n_cols = 2
        n_rows = (n_methods + 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4 * n_rows), squeeze=False)

        for idx, (method, hv_values) in enumerate(methods_hv.items()):
            row, col = divmod(idx, n_cols)
            ax = axes[row][col]
            ax.plot(
                hv_values,
                #color=colors[idx % len(colors)],
                color=colors[idx],
                linewidth=2
            )
            ax.set_title(f"Hypervolume - {method}")
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Hypervolume")
            ax.grid(True)

        # Hide any unused subplots
        for idx in range(len(methods_hv), n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row][col].axis("off")

        plt.tight_layout()
        plt.show()

    else:
        plt.figure(figsize=(6, 5))
        for i, (method, hv_values) in enumerate(methods_hv.items()):
            plt.plot(
                hv_values,
                color=colors[i % len(colors)],
                label=method,
                linewidth=2
            )

        plt.xlabel("Iterations")
        plt.ylabel("Hypervolume")
        plt.title("Hypervolume Evolution Across Methods")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def plot_pareto_2d_multi(
    pareto_fronts: Dict[str, List[List[float]]],
    colors: List[str] = None,
    markers: List[str] = None,
    title: str = "Pareto Front Comparison (2D)"
) -> None:
    """
    Plots multiple Pareto frontiers on a single 2D plot.

    Args:
        pareto_fronts: Dictionary where keys are method names and values are lists of 2D objective vectors.
        colors: Optional list of colors for each method.
        markers: Optional list of marker styles for each method.
        title: Plot title.
    """
    if colors is None:
        colors = plt.cm.tab10.colors
    if markers is None:
        markers = ['o', 's', 'v', 'D', '^', 'x', '*', 'P', 'X']

    plt.figure(figsize=(8, 6))
    for i, (method, front) in enumerate(pareto_fronts.items()):
        front = np.array(front)
        plt.scatter(
            front[:, 0], front[:, 1],
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            label=method,
            s=60,
            edgecolors='black'
        )

    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_pareto_2d_multiple(pareto_dict: Dict[str, np.ndarray], 
                            show_subplot: bool = True,
                            title: str = "Pareto Frontier (2D)",
                            x_label: str = "Objective 1",
                            y_label: str = "Objective 2",
                            point: Optional[np.ndarray] = None,
                            colors: Optional[str] = None,
                            ) -> None:
    """
    Plot multiple 2D Pareto frontiers either side by side or in a single plot.

    Parameters:
        pareto_dict (Dict[str, np.ndarray]): Dictionary of method names to 2D objective arrays.
        show_subplot (bool): If True, shows subplots per method. If False, all in one plot.
        title: Plot title.
        point: Optional np.array with shape (2,) to highlight on the plot.
    """
    #colors = plt.cm.tab10.colors

    if show_subplot:  
        n_methods = len(pareto_dict)
        fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))

        if n_methods == 1:
            axes = [axes]

        i = 0
        #colors = ['#FF5C8D', '#4BD6A0', '#9D5CFF']
        #colors = ['#FF5C8D', '#FF5C8D', '#FF5C8D']
        for ax, (method, objs) in zip(axes, pareto_dict.items()):
            if point is not None:
                point = np.asarray(point).flatten()
                ax.scatter([point[0]], [point[1]], color="#FFC145", s=30, label="Baseline", edgecolors='black', linewidths=0.5)
            #ax.scatter(objs[:, 0], objs[:, 1], label=method, s=30, color=colors[i % len(colors)])
            ax.scatter(objs[:, 0], objs[:, 1], label=method, s=30, color=colors[i], edgecolors='black', linewidths=0.5)
            i = i + 1
            ax.set_title(f"{method}")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.grid(True)

        plt.tight_layout()
        plt.show()
    else:
        colors = ['#FF5C8D', '#9D5CFF', '#4A6BFF']
        plt.figure(figsize=(6, 5))
        i = 0
        for method, objs in pareto_dict.items():
            if point is not None:
                point = np.asarray(point).flatten()
                plt.scatter([point[0]], [point[1]], color="indianred", s=30, label="Baseline",  edgecolors='black', linewidths=0.5)
            plt.scatter(objs[:, 0], objs[:, 1], label=method, s=30, color=colors[i], edgecolors='black', linewidths=0.5)
            i = i + 1
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def plot_pareto_3d_multiple(pareto_dict: Dict[str, np.ndarray], show_subplot: bool = True) -> None:
    """
    Plot multiple 3D Pareto frontiers either side by side or in a single plot.

    Parameters:
        pareto_dict (Dict[str, np.ndarray]): Dictionary of method names to 3D objective arrays.
        show_subplot (bool): If True, shows subplots per method. If False, all in one plot.
    """
    if show_subplot:
        n_methods = len(pareto_dict)
        fig = plt.figure(figsize=(6 * n_methods, 6))

        for i, (method, objs) in enumerate(pareto_dict.items()):
            ax = fig.add_subplot(1, n_methods, i + 1, projection='3d')
            ax.scatter(objs[:, 0], objs[:, 1], objs[:, 2], label=method)
            ax.set_title(f"{method}")
            ax.set_xlabel("Objective 1")
            ax.set_ylabel("Objective 2")
            ax.set_zlabel("Objective 3")

        plt.tight_layout()
        plt.show()
    else:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        for method, objs in pareto_dict.items():
            ax.scatter(objs[:, 0], objs[:, 1], objs[:, 2], label=method)

        ax.set_title("Pareto Frontiers (3D)")
        ax.set_xlabel("Objective 1")
        ax.set_ylabel("Objective 2")
        ax.set_zlabel("Objective 3")
        ax.legend()
        plt.tight_layout()
        plt.show()

def plot_pareto_3d_multiple_view(
    pareto_dict: Dict[str, np.ndarray],
    show_subplot: bool = True,
    n_views: int = 2
) -> None:
    """
    Plot multiple 3D Pareto frontiers with optional multiple views per method.

    Parameters:
        pareto_dict (Dict[str, np.ndarray]): Dictionary mapping method names to 3D objective arrays.
        show_subplot (bool): Whether to show subplots (True) or a single combined plot (False).
        n_views (int): Number of different perspectives to show per method (e.g., 2, 3).
    """
    if show_subplot:
        n_methods = len(pareto_dict)
        fig = plt.figure(figsize=(6 * n_views, 5 * n_methods))

        # Gera n_views ângulos distintos
        azims = np.linspace(45, 360, n_views, endpoint=False)
        elevs = np.linspace(20, 60, n_views, endpoint=True)
        view_angles = list(zip(elevs, azims))

        for row_i, (method, objs) in enumerate(pareto_dict.items()):
            for view_i in range(n_views):
                elev, azim = view_angles[view_i]
                ax = fig.add_subplot(n_methods, n_views, row_i * n_views + view_i + 1, projection='3d')
                ax.scatter(objs[:, 0], objs[:, 1], objs[:, 2])
                ax.set_title(f"{method} (View {view_i + 1})")
                ax.set_xlabel("Objective 1")
                ax.set_ylabel("Objective 2")
                ax.set_zlabel("Objective 3")
                ax.view_init(elev=elev, azim=azim)

        plt.tight_layout()
        plt.show()
    else:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        for method, objs in pareto_dict.items():
            ax.scatter(objs[:, 0], objs[:, 1], objs[:, 2], label=method)
        ax.set_title("Pareto Frontiers (3D)")
        ax.set_xlabel("Objective 1")
        ax.set_ylabel("Objective 2")
        ax.set_zlabel("Objective 3")
        ax.legend()
        plt.tight_layout()
        plt.show()

def plot_pareto_3d_multiples(
    pareto_dict: Dict[str, npt.NDArray[np.float64]],
    x_label: str = "Objective 1",
    y_label: str = "Objective 2",
    z_label: str = "Objective 3",
    point: Optional[npt.NDArray[np.float64]] = None,
    colors: Optional[str] = None,
    figsize: tuple = (10, 5)
) -> None:
    """
    Plot multiple 3D Pareto frontiers with two fixed perspectives per method (stacked subplots).

    Parameters:
    - pareto_dict: Dictionary mapping method name to 3D objective array.
    - figsize: Size of the figure (width, height per row).
    """
    n_methods = len(pareto_dict)
    fig = plt.figure(figsize=(figsize[0], figsize[1] * n_methods))

    views = [(10, -120), (20, -20)]  # Two preferred perspectives
    idx = 1

    i = 0
    for method, objs in pareto_dict.items():
        for view_i, (elev, azim) in enumerate(views):
            ax = fig.add_subplot(n_methods, 2, idx, projection='3d')
            if point is not None:
                point = np.asarray(point).flatten()
                ax.scatter([point[0]], [point[1]], [point[2]], color="#FFC145", label="Baseline", s=5, edgecolors='black', linewidths=0.5)
            ax.scatter(objs[:, 0], objs[:, 1], objs[:, 2], label=method, color=colors[i], edgecolors='black', linewidths=0.5, marker='.')
            ax.view_init(elev, azim)
            ax.set_title(f"{method} - View {view_i + 1}")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_zlabel(z_label)
            idx += 1
        i += 1

    plt.tight_layout()
    plt.show()

def plot_parallel_coordinates_all(
    pareto_dict: Dict[str, Any],
    normalize: bool = True,
    same_plot: bool = False,
    figsize: tuple = (15, 5),
    labels: Optional[List[str]] = None,
    title: str = "Parallel Coordinates (All Methods)",
    colormap: str = "viridis",
    colors: Optional[str] = None,
    alpha: float = 1.0,
) -> None:
    """
    Plots parallel coordinate plots for multiple pareto_dict.

    Parameters
    ----------
    pareto_dict : dict
        Dictionary with method names as keys and solver objects as values.
    normalize : bool
        Whether to normalize the objectives between 0 and 1 for visualization.
    same_plot : bool
        If True, plots all methods in the same plot. If False, uses subplots.
    figsize : tuple
        Size of the figure for subplots.
    """
    dfs = []

    num_objectives = len(pareto_dict["mola"][0])
    default_labels = [f"Objective {i+1}" for i in range(num_objectives)]
    columns = labels if labels else default_labels

    for method_name, objectives in pareto_dict.items():
        objs = objectives
        #if hasattr(solver, 'solutions_list'):
        #    objs = np.array([s.objs for s in solver.solutions_list])
        #elif hasattr(solver, 'solutionsList'):
        #    objs = np.array([s.objs for s in solver.solutionsList])
        #else:
        #    raise ValueError(f"Solver '{method_name}' does not contain recognizable solution list attribute.")

        if normalize:
            objs = (objs - objs.min(axis=0)) / (objs.max(axis=0) - objs.min(axis=0) + 1e-8)

        df = pd.DataFrame(objs, columns=columns)
        df['method'] = method_name
        dfs.append(df)

    all_data = pd.concat(dfs, ignore_index=True)

    if same_plot:
        plt.figure(figsize=figsize)
        parallel_coordinates(all_data, 'method', colormap=plt.cm.Set1, alpha=alpha)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        methods = list(pareto_dict.keys())
        n = len(methods)
        fig, axes = plt.subplots(1, n, figsize=(figsize[0] * n / 3, figsize[1]), sharey=True)

        if n == 1:
            axes = [axes]  # Ensure it's iterable

        for i, method in enumerate(methods):
            if colors is None:
                colors = [sns.color_palette()[i]]
            subset = all_data[all_data['method'] == method]
            parallel_coordinates(subset, 'method', ax=axes[i], color=colors[i])
            axes[i].set_title(method)
            axes[i].grid(True)
        plt.tight_layout()
        plt.show()

def plot_pareto_2d_set_limit_one_line(pareto_dict: Dict[str, npt.NDArray[np.float64]], 
                            show_subplot: bool = True,
                            title: str = "Pareto Frontier (2D)",
                            x_label: str = "Objective 1",
                            y_label: str = "Objective 2",
                            point: Optional[npt.NDArray[np.float64]] = None,
                            colors: Optional[str] = None,
                            axis_mode: str = "global",  # "first" ou "global"
                            ) -> None:
    n_methods = len(pareto_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))

    if n_methods == 1:
        axes = [axes]

    # Definindo os limites dos eixos
    if axis_mode == "first":
        first_objs = next(iter(pareto_dict.values()))
        x_min, x_max = first_objs[:, 0].min(), first_objs[:, 0].max()
        y_min, y_max = first_objs[:, 1].min(), first_objs[:, 1].max()
    elif axis_mode == "global":
        all_points = np.vstack(list(pareto_dict.values()))
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    else:
        raise ValueError("axis_mode must be 'first' or 'global'")

    if colors is None:
        colors = ['#FF5C8D', '#4BD6A0', '#9D5CFF', '#FFC145', '#56B4E9']
    
    for i, (ax, (method, objs)) in enumerate(zip(axes, pareto_dict.items())):
        if point is not None:
            point = np.asarray(point).flatten()
            ax.scatter([point[0]], [point[1]], color="#FFC145", s=30, label="Baseline", edgecolors='black', linewidths=0.5)
        ax.scatter(objs[:, 0], objs[:, 1], label=method, s=30, color=colors[i % len(colors)], edgecolors='black', linewidths=0.5)
        ax.set_title(f"{method}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

from math import ceil

def plot_pareto_2d_set_limit(
    pareto_dict: Dict[str, npt.NDArray[np.float64]], 
    show_subplot: bool = True,
    title: str = "Pareto Frontier (2D)",
    x_label: str = "Objective 1",
    y_label: str = "Objective 2",
    point: Optional[npt.NDArray[np.float64]] = None,
    colors: Optional[str] = None,
    axis_mode: str = "global"  # "first" or "global"
) -> None:
    n_methods = len(pareto_dict)

    # Define layout: 1 row if <= 3 methods, otherwise 2 rows
    if n_methods <= 3:
        nrows, ncols = 1, n_methods
    else:
        nrows, ncols = 2, ceil(n_methods / 2)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    # Eixos fixos conforme solicitado
    if axis_mode == "first":
        first_objs = next(iter(pareto_dict.values()))
        x_min, x_max = first_objs[:, 0].min(), first_objs[:, 0].max()
        y_min, y_max = first_objs[:, 1].min(), first_objs[:, 1].max()
    elif axis_mode == "global":
        all_points = np.vstack(list(pareto_dict.values()))
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    else:
        raise ValueError("axis_mode must be 'first' or 'global'")

    if colors is None:
        colors = ['#FF5C8D', '#4BD6A0', '#9D5CFF', '#FFC145', '#56B4E9']

    for i, (method, objs) in enumerate(pareto_dict.items()):
        ax = axes[i]
        if point is not None:
            point = np.asarray(point).flatten()
            ax.scatter([point[0]], [point[1]], color="#FFC145", s=30, label="Baseline",
                       edgecolors='black', linewidths=0.5)
        ax.scatter(objs[:, 0], objs[:, 1], label=method, s=30,
                   color=colors[i % len(colors)], edgecolors='black', linewidths=0.5)
        ax.set_title(f"{method}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True)

    # Esconde eixos extras (se houver mais subplots que métodos)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional

def plot_pareto_3d_set_limit(
    pareto_dict: Dict[str, np.ndarray],
    x_label: str = "Objective 1",
    y_label: str = "Objective 2",
    z_label: str = "Objective 3",
    point: Optional[np.ndarray] = None,
    colors: Optional[str] = None,
    figsize: tuple = (10, 5),
    axis_mode: str = "first"  # "first" ou "global"
) -> None:
    """
    Plot multiple 3D Pareto frontiers with fixed perspectives and synchronized axes.

    Parameters:
    - pareto_dict: Dictionary mapping method name to 3D objective array.
    - axis_mode: 'first' uses limits from first method; 'global' uses global min/max for all.
    """
    n_methods = len(pareto_dict)
    fig = plt.figure(figsize=(figsize[0], figsize[1] * n_methods))

    views = [(10, -120), (20, -20)]  # Two preferred perspectives
    idx = 1

    # Cores padrão se não especificado
    if colors is None:
        colors = ['#FF5C8D', '#4BD6A0', '#9D5CFF', '#FFC145', '#56B4E9']

    # Definindo limites dos eixos
    if axis_mode == "first":
        first_objs = next(iter(pareto_dict.values()))
        x_min, x_max = first_objs[:, 0].min(), first_objs[:, 0].max()
        y_min, y_max = first_objs[:, 1].min(), first_objs[:, 1].max()
        z_min, z_max = first_objs[:, 2].min(), first_objs[:, 2].max()
    elif axis_mode == "global":
        all_points = np.vstack(list(pareto_dict.values()))
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
        z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()
    else:
        raise ValueError("axis_mode must be 'first' or 'global'")

    i = 0
    for method, objs in pareto_dict.items():
        for view_i, (elev, azim) in enumerate(views):
            ax = fig.add_subplot(n_methods, 2, idx, projection='3d')
            if point is not None:
                point = np.asarray(point).flatten()
                ax.scatter([point[0]], [point[1]], [point[2]], color="#FFC145", label="Baseline", s=5, edgecolors='black', linewidths=0.5)
            ax.scatter(objs[:, 0], objs[:, 1], objs[:, 2], label=method,
                       color=colors[i % len(colors)], edgecolors='black', linewidths=0.5, marker='.')
            ax.view_init(elev, azim)
            ax.set_title(f"{method} - View {view_i + 1}")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_zlabel(z_label)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            idx += 1
        i += 1

    plt.tight_layout()
    plt.show()
