import pandas as pd
import numpy as np
import numpy.typing as npt
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pandas.plotting import parallel_coordinates
from typing import Any
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D


# ------------------ main ------------------
def plot_pareto(
    methods: dict,
    alpha: float = 1.0,
    subset: npt.NDArray[np.float64] | list[float] = np.array([]),
    save_path: str  = "",
    labels: tuple | list[str] = (),
    color: list[str] | str = ['#d73027', '#fc8d59', '#4575b4', '#fee090', '#91bfdb', '#e0f3f8'],
    subset_label: str | list[str] = "Baseline Model",
    title: str = "Pareto Frontier",
    fontsize: int = 20,
    figsize: tuple[int, int] = (10, 7),
    s_factor: int = 2,
    limits: list = [],
    subset_marker: str | list[str] = "X",
    x_threshold: int | float | None = None,
    y_threshold: int | float | None = None,
    threshold_color: str = "#313695"
) -> None:
    """
    Plots the Pareto frontiers for different methods using Matplotlib.

    Displays either 2D, 3D, or parallel coordinates (>= 4D) plots depending on the
    number of objectives. Supports highlighting a reference subset and saving the figure.

    Args:
        methods (dict): Dictionary where keys are method names and values are objective arrays.
        alpha (float): Transparency of the scatter points.
        subset (np.ndarray): A reference subset to be highlighted.
        save_path (str): Optional path to save the figure.
        labels (list[str] or tuple): Labels for each objective axis.
        color (list[str]): List of colors to use per method.
        subset_label (str): Label for the reference subset.
        title (str): Title for the plot.
        fontsize (int): Font size for labels and title.
        figsize (tuple[int, int]): Size of each subplot.
        s_factor (int): Scaling factor for scatter subset sizes.
    """
    if subset is None: subset = []
    if len(subset) > 0 and not isinstance(subset[0], list): subset = [subset]
    if isinstance(color, str): color = [color]

    all_data = np.concatenate(list(methods.values()), axis=0)
    num_objectives = all_data.shape[1]
    labels = labels if len(labels) != 0 else [f"Objective {i+1}" for i in range(num_objectives)]
    limits = limits if len(limits) > 0 else _get_limits(all_data)

    nrows, ncols, gs, positions = _get_layout(len(methods), figsize)
    fig = plt.figure(figsize=(ncols * figsize[0], nrows * figsize[1]))
    gs = gridspec.GridSpec(nrows, ncols, figure=fig)

    for idx, (key, values) in enumerate(methods.items()):
        row, col = positions[idx]
        ax = fig.add_subplot(gs[row, col], projection="3d" if values.shape[1] == 3 else None)

        if num_objectives < 4:
            _plot_2d_or_3d(
                ax=ax,
                values=values,
                key=key,
                labels=labels,
                c=color[idx % len(color)],
                alpha=alpha,
                s_factor=s_factor,
                fontsize=fontsize,
                limits=limits,
                x_threshold=x_threshold,
                y_threshold=y_threshold,
                threshold_color=threshold_color
            )
            if len(subset) > 0:
                _plot_subset(
                    ax=ax,
                    subset=subset,
                    values=values,
                    subset_marker=subset_marker,
                    subset_label=subset_label,
                    s_factor=s_factor,
                    fontsize=fontsize
                )
        else:
            _plot_parallel(
                ax=ax,
                values=values,
                key=key,
                labels=labels,
                color=color[idx % len(color)],
                alpha=alpha,
                s_factor=s_factor,
                subset=subset,
                subset_label=subset_label,
                y_threshold=y_threshold,
                threshold_color=threshold_color
            )

    # delete extra subplots
    total_subplots = nrows * ncols
    for idx in range(len(methods), total_subplots):
        fig.add_subplot(gs[positions[idx][0], positions[idx][1]]).axis("off")

    plt.grid(True)
    fig.tight_layout()
    if save_path != "": 
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

# ------------------ helpers ------------------

def _get_layout(num_plots: int, figsize: tuple[int, int]) -> tuple[int, int, gridspec.GridSpec, list[tuple[int, int]]]:
    """Compute subplot layout (rows, cols, positions)."""
    layout_configs = {
        1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2),
        5: (2, 3), 6: (2, 3), 7: (3, 3), 8: (3, 3), 9: (3, 3),
    }
    nrows, ncols = layout_configs.get(
        num_plots,
        (
            int(np.ceil(np.sqrt(num_plots))),
            int(np.ceil(num_plots / np.ceil(np.sqrt(num_plots)))),
        ),
    )
    fig = plt.figure(figsize=(ncols * figsize[0], nrows * figsize[1]))
    gs = gridspec.GridSpec(nrows, ncols, figure=fig)
    positions = [(i, j) for i in range(nrows) for j in range(ncols)]
    return nrows, ncols, gs, positions

def _get_limits(all_data: np.ndarray) -> list[tuple[float, float]]:
    """Compute axis limits with margins."""
    return [
        (
            all_data[:, i].min() - all_data[:, i].std(),
            all_data[:, i].max() + all_data[:, i].std(),
        )
        for i in range(all_data.shape[1])
    ]

def _plot_subset(
    ax: Axes | Axes3D,
    subset: npt.NDArray[np.float64] | list[float],
    values: npt.NDArray[np.float64],
    subset_marker: str | list[str],
    subset_label: str | list[str],
    s_factor: float,
    fontsize: int,
) -> None:
    """Plot reference subset points (2D or 3D)."""
    subset_array = np.atleast_2d(np.array(subset))
    if isinstance(subset_marker, str):
        subset_marker = [subset_marker] * len(subset)
    if isinstance(subset_label, str):
        subset_label = [subset_label]

    

    for i, p in enumerate(subset_array):
        kwargs = dict(
            marker=subset_marker[i],
            c="black", edgecolors="k",
            s=70 * s_factor,
            linewidth=1.5 * (s_factor // 2),
            label=subset_label[i] if i == 0 or len(subset_label) > 1 else None,
        )
        if values.shape[1] == 2:
            ax.scatter(x=p[0], y=p[1], **kwargs)
        else:
            ax.scatter(xs=p[0], ys=p[1], zs=p[2], **kwargs)

    ax.legend(fontsize=fontsize - 5)

def _plot_2d_or_3d(
    ax: Any,
    values: np.ndarray,
    key: str,
    labels: list[str] | tuple,
    c: str | list[str],
    alpha: float,
    s_factor: float,
    fontsize: int,
    limits: list[tuple[float, float]],
    x_threshold: int | float | None = None,
    y_threshold: int | float | None = None,
    threshold_color: str = "#313695"
) -> None:
    """Handle 2D/3D scatter plots."""
    rotation_y=90
    if values.shape[1] == 2:
        ax.scatter(
            x=values[:, 0], y=values[:, 1],
            alpha=alpha, c=c,
            label=f"{key.upper().replace('_', ' ')} Solutions",
            edgecolor="k", linewidth=1.5 * (s_factor // 2),
            s=50 * s_factor,
        )
    else:
        ax.scatter(
            xs=values[:, 0], ys=values[:, 1], zs=values[:, 2],
            alpha=alpha, c=c,
            label=f"{key.upper().replace('_', ' ')} Solutions",
            edgecolor="k", linewidth=1.5 * (s_factor // 2),
            s=50 * s_factor,
        )
        ax.set_zlabel(labels[2].capitalize(), fontsize=fontsize, labelpad=1, rotation=90)
        ax.tick_params(axis="z", labelsize=fontsize - 5, pad=3.5)
        ax.set_zlim(limits[2])
        rotation_y = 45

    if x_threshold is not None:
        plt.axvline(x=x_threshold, color=threshold_color, linestyle="--", linewidth=3, label="Threshold")
    if y_threshold is not None:
        plt.axhline(y=y_threshold, color=threshold_color, linestyle="--", linewidth=3, label="Threshold")

    ax.set_xlabel(labels[0].capitalize(), fontsize=fontsize, labelpad=8)
    ax.set_ylabel(labels[1].capitalize(), fontsize=fontsize,  labelpad=6, rotation=rotation_y)
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.legend(fontsize=fontsize, loc="upper right")
    ax.tick_params(axis="x", labelsize=fontsize - 5, pad=1) 
    ax.tick_params(axis="y", labelsize=fontsize - 5, pad=1)
    ax.grid()

def _plot_parallel(
    ax: Axes,
    values: np.ndarray,
    key: str,
    labels: list[str] | tuple,
    color: str | list[str],
    alpha: float,
    s_factor: float,
    subset: np.ndarray | list[np.ndarray] | list[float],
    subset_label: str | list[str],
    y_threshold: int | float | None = None,
    threshold_color: str = "#313695"
) -> None:
    """Handle parallel coordinates plot (>=4 objectives)."""
    df = pd.DataFrame(values, columns=labels)
    df["Method"] = f"{key.upper()} Solutions"

    # normalize
    df.iloc[:, :-1] = (df.iloc[:, :-1] - df.iloc[:, :-1].min()) / (
        df.iloc[:, :-1].max() - df.iloc[:, :-1].min()
    )

    parallel_coordinates(df, class_column="Method", color=color, alpha=alpha, ax=ax, linewidth=1.5)
    
    if y_threshold is not None:
        plt.axhline(y=y_threshold, color=threshold_color, linestyle="--", linewidth=3, label="Threshold")

    if subset is not None and len(subset) != 0:
        df_subset = pd.DataFrame(np.array(subset), columns=labels)

        if isinstance(subset_label, list):
            df_subset["Solution"] = subset_label
        else:
            df_subset["Solution"] = [subset_label] * len(subset)

        parallel_coordinates(
            df_subset, class_column="Solution", color=["#a50026", "#f46d43", '#fdae61'],
            alpha=1.0, linewidth=1.5 * (s_factor // 2), ax=ax
        )

    ax.set_xticklabels(df.columns[:-1], rotation=45, fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    plt.xlabel("Objectives", fontsize=20)
    plt.ylabel("Values", fontsize=20)


### PARETO FRONTIER WITH PLOTLY
# ------------------ main ------------------
def plot_pareto_plotly(
    methods: dict[Any, Any],
    point: npt.NDArray[np.float64] = np.array([]),
    save_path: str = "",
    labels: tuple | list[str] = (),
    color: list[str] = ["#FF5C8D", "#4BD6A0", "#9D5CFF"],
    point_label: str = "Baseline Model",
    title: str = "Pareto Frontier",
    fontsize: int = 20,
    figsize: tuple[int, int] = (10, 7),
) -> None:
    """
    Selects the appropriate Plotly visualization (2D, 3D, or parallel coordinates (>= 4D))
    based on the number of objectives.

    Args:
        methods (dict): Dictionary of method names to arrays of objectives.
        point (np.ndarray): Optional reference point to plot.
        save_path (str): Optional path to save HTML output.
        labels (list[str] or tuple): Objective names.
        color (list[str]): Colors for each method.
        point_label (str): Name for the reference point.
        title (str): Plot title.
        fontsize (int): Font size.
        figsize (tuple[int, int]): Figure dimensions.

    Returns:
        None
    """
    first_key = next(iter(methods))
    num_objectives = methods[first_key].shape[1]

    if num_objectives == 2:
        plot_pareto_plotly_2d(
            methods=methods,
            point=point,
            save_path=save_path,
            labels=labels,
            color=color,
            point_label=point_label,
            title=title,
            fontsize=fontsize,
            figsize=figsize,
        )
    elif num_objectives == 3:
        plot_pareto_plotly_3d(
            methods=methods,
            point=point,
            save_path=save_path,
            labels=labels,
            color=color,
            point_label=point_label,
            title=title,
            fontsize=fontsize,
            figsize=figsize,
        )
    else:
        plot_pareto_plotly_4d(
            methods=methods,
            point=point,
            save_path=save_path,
            labels=labels,
            color=color,
            point_label=point_label,
            title=title,
            fontsize=fontsize,
            figsize=figsize,
        )

# ------------------ helpers ------------------
def plot_pareto_plotly_2d(
    methods: dict[Any, Any],
    point: npt.NDArray[np.float64] = np.array([]),
    save_path: str = "",
    labels: tuple | list[str] = (),
    color: list[str] = ["#FF5C8D", "#4BD6A0", "#9D5CFF"],
    point_label: str = "Baseline Model",
    title: str = "Pareto Frontier",
    fontsize: int = 20,
    figsize: tuple[int, int] = (10, 7),
) -> None:
    """
    Generates 2D Pareto subplots using Plotly.

    Args:
        methods (dict): Dictionary of method names to 2D objective arrays.
        point (np.ndarray): Optional reference point.
        save_path (str): Path to save the plot (HTML).
        labels (list[str] or tuple): Axis labels.
        color (list[str]): Marker colors.
        point_label (str): Label for the point.
        title (str): Title of the plot.
        fontsize (int): Font size.
        figsize (tuple[int, int]): Figure size.

    Returns:
        None
    """
    num_plots: int = len(methods)

    layout_configs: dict[int, tuple[int, int]] = {
        1: (1, 1),
        2: (1, 2),
        3: (1, 3),
        4: (2, 2),
        5: (2, 3),
        6: (2, 3),
        7: (3, 3),
        8: (3, 3),
        9: (3, 3),
    }

    rows, cols = layout_configs[num_plots]
    positions = [(i, j) for i in range(rows) for j in range(cols)]

    first_key = next(iter(methods))
    num_objectives = methods[first_key].shape[1]
    default_labels = [f"Objective {i+1}" for i in range(num_objectives)]
    labels = labels if len(labels) != 0 else default_labels

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=list(methods.keys()),
        shared_xaxes=True,
        shared_yaxes=True,
    )

    for i, ((row, col), (key, values)) in enumerate(zip(positions, methods.items())):
        fig.add_trace(
            go.Scatter(
                x=values[:, 0],
                y=values[:, 1],
                mode="markers",
                marker=dict(
                    size=12,
                    color=color[i % len(color)],
                    line=dict(color="black", width=1),
                ),
            ),
            row=row + 1,
            col=col + 1,
        )

        if len(point) != 0:
            fig.add_trace(
                go.Scatter(
                    x=[point[0]],
                    y=[point[1]],
                    mode="markers+text",
                    name=point_label,
                    marker=dict(size=15, color="black", symbol="star"),
                    text=[point_label],
                    textposition="top right",
                ),
                row=row + 1,
                col=col + 1,
            )

    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=fontsize, color="black")

    fig.update_layout(
        height=int(figsize[1] * 100),
        width=int(figsize[0] * 100),
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=fontsize)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=fontsize, color="black"),
        margin=dict(l=60, r=40, t=80, b=60),
        showlegend=False,
    )

    # Update all x and y axes titles
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            fig.update_xaxes(
                title_text=labels[0],
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
                showgrid=True,
                gridcolor="lightgray",
                gridwidth=1,
                zeroline=False,
                row=i,
                col=j,
            )
            fig.update_yaxes(
                title_text=labels[1] if ((i == 1) and (j == 1)) else None,
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
                showgrid=True,
                gridcolor="lightgray",
                gridwidth=1,
                zeroline=False,
                row=i,
                col=j,
            )

    fig.show()

    if save_path != "":
        fig.write_html(save_path)


def plot_pareto_plotly_3d(
    methods: dict[Any, Any],
    point: npt.NDArray[np.float64] = np.array([]),
    save_path: str = "",
    labels: tuple | list[str] = (),
    color: list[str] = ["#FF5C8D", "#4BD6A0", "#9D5CFF"],
    point_label: str = "Baseline Model",
    title: str = "Pareto Frontier",
    fontsize: int = 20,
    figsize: tuple[int, int] = (10, 7),
) -> None:
    """
    Plots 3D Pareto frontiers using Plotly with subplot support.

    Args:
        methods (dict): Dictionary with 3D objective data per method.
        point (np.ndarray): Optional 3D point to mark.
        save_path (str): Optional HTML file path to save.
        labels (list[str] or tuple): Axis labels.
        color (list[str]): Colors per method.
        point_label (str): Point label.
        title (str): Plot title.
        fontsize (int): Font size.
        figsize (tuple[int, int]): Plot size.

    Returns:
        None
    """
    num_plots = len(methods)

    layout_configs = {
        1: (1, 1),
        2: (1, 2),
        3: (1, 3),
        4: (2, 2),
        5: (2, 3),
        6: (2, 3),
        7: (3, 3),
        8: (3, 3),
        9: (3, 3),
    }

    rows, cols = layout_configs[num_plots]
    positions = [(i, j) for i in range(rows) for j in range(cols)]

    first_key = next(iter(methods))
    num_objectives = methods[first_key].shape[1]
    default_labels = [f"Objective {i+1}" for i in range(num_objectives)]
    labels = labels if len(labels) != 0 else default_labels

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=list(methods.keys()),
        specs=[[{"type": "scene"} for _ in range(cols)] for _ in range(rows)],
    )

    for i, ((row, col), (key, values)) in enumerate(zip(positions, methods.items())):
        scene_id = f"scene{(row * cols + col + 1)}"
        fig.add_trace(
            go.Scatter3d(
                x=values[:, 0],
                y=values[:, 1],
                z=values[:, 2],
                mode="markers",
                marker=dict(
                    size=12,
                    color=color[i % len(color)],
                    line=dict(color="black", width=1),
                ),
            ),
            row=row + 1,
            col=col + 1,
        )

        if len(point) != 0:
            fig.add_trace(
                go.Scatter3d(
                    x=[point[0]],
                    y=[point[1]],
                    z=[point[2]],
                    mode="markers+text",
                    name=point_label,
                    marker=dict(size=15, color="black", symbol="cross"),
                    text=[point_label],
                    textposition="top right",
                ),
                row=row + 1,
                col=col + 1,
            )

        fig.update_layout(
            {
                scene_id: dict(
                    xaxis_title=labels[0],
                    yaxis_title=labels[1],
                    zaxis_title=labels[2],
                    xaxis=dict(showgrid=True, gridcolor="lightgray"),
                    yaxis=dict(showgrid=True, gridcolor="lightgray"),
                    zaxis=dict(showgrid=True, gridcolor="lightgray"),
                )
            }
        )

    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=fontsize, color="black")

    fig.update_layout(
        height=int(figsize[1] * 100),
        width=int(figsize[0] * 100),
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=fontsize)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=fontsize, color="black"),
        margin=dict(l=60, r=40, t=80, b=60),
        showlegend=False,
    )

    fig.show()

    if save_path != "":
        fig.write_html(save_path)


def plot_pareto_plotly_4d(
    methods: dict[Any, Any],
    point: npt.NDArray[np.float64] = np.array([]),
    save_path: str = "",
    labels: tuple | list[str] = (),
    color: list[str] = ["#FF5C8D", "#4BD6A0", "#9D5CFF"],
    point_label: str = "Baseline Model",
    title: str = "Pareto Frontier",
    fontsize: int = 20,
    figsize: tuple[int, int] = (10, 7),
) -> None:
    """
    Displays 4D+ Pareto solutions using Plotly's parallel coordinates.

    Args:
        methods (dict): Dictionary with objective arrays.
        point (np.ndarray): Ignored in 4D plot.
        save_path (str): Optional save path.
        labels (list[str] or tuple): Labels for each dimension.
        color (list[str]): Colors for each method.
        point_label (str): Label for the point (unused).
        title (str): Plot title.
        fontsize (int): Font size for all text.
        figsize (tuple[int, int]): Plot size.

    Returns:
        None
    """
    num_plots = len(methods)

    layout_configs = {
        1: (1, 1),
        2: (1, 2),
        3: (1, 3),
        4: (2, 2),
        5: (2, 3),
        6: (2, 3),
        7: (3, 3),
        8: (3, 3),
        9: (3, 3),
    }

    rows, cols = layout_configs[num_plots]
    positions = [(i, j) for i in range(rows) for j in range(cols)]

    first_key = next(iter(methods))
    num_objectives = methods[first_key].shape[1]

    default_labels = [f"Objective {i+1}" for i in range(num_objectives)]
    labels = labels if len(labels) != 0 else default_labels

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=list(methods.keys()),
        specs=[[{"type": "domain"} for _ in range(cols)] for _ in range(rows)],
    )

    for i, ((row, col), (key, values)) in enumerate(zip(positions, methods.items())):
        dimensions = [
            dict(label=labels[d], values=values[:, d]) for d in range(values.shape[1])
        ]

        fig.add_trace(
            go.Parcoords(line=dict(color=color[i % len(color)]), dimensions=dimensions),
            row=row + 1,
            col=col + 1,
        )

    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=fontsize, color="black")

    fig.update_layout(
        height=int(figsize[1] * 100),
        width=int(figsize[0] * 100),
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=fontsize)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=fontsize, color="black"),
        margin=dict(l=60, r=40, t=80, b=60),
        showlegend=False,
    )

    fig.show()

    if save_path != "":
        fig.write_html(save_path)

### HYPERVOLUME
def plot_hypervolume(
    hypervolume_values: list[float], 
    method: str = "", 
    color: str = "tab:pink"
) -> None:
    """
    Plots the evolution of hypervolume over iterations.

    Args:
        hypervolume_values (list[float]): List of hypervolume values.
        method (str): Optional method name for labeling.
        color (str): Line color.

    Returns:
        None
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
    methods_hv: dict[Any, Any],
    subplots: bool = False,
    save_path: str = "",
    colors: list[str] | str = ['#d73027', '#fc8d59', '#4575b4', '#fee090', '#91bfdb', '#e0f3f8'],
) -> None:
    """
    Plots hypervolume evolution for multiple methods, optionally in subplots.

    Args:
        methods_hv (dict): Method names mapped to hypervolume value lists.
        subplots (bool): Whether to plot in subplots.
        save_path (str): Optional path to save the figure.
        colors (list[str]): Color list for lines.

    Returns:
        None
    """
    if isinstance(colors, str):
        colors = [colors]

    if subplots:
        n_methods = len(methods_hv)
        n_cols = 2
        n_rows = (n_methods + 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(10, 4 * n_rows), squeeze=False
        )

        for idx, (method, hv_values) in enumerate(methods_hv.items()):
            row, col = divmod(idx, n_cols)
            ax = axes[row][col]
            ax.plot(
                hv_values,
                # color=colors[idx % len(colors)],
                color=colors[idx],
                linewidth=2,
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
        markers = ["o", "s", "D", "^", "v", "*", "X", "P"]
        plt.figure(figsize=(6, 5))
        for i, (method, hv_values) in enumerate(methods_hv.items()):
            color= colors[i % len(colors)]
            label = method.upper().replace("_", " ")
            plt.plot(
                hv_values, 
                color=color, 
                linewidth=2,
                #label=label
            )
            plt.plot(
                len(hv_values) - 1,
                hv_values[-1],
                color=color,
                marker=markers[i % len(markers)],
                markersize=8,
                linestyle="solid",
                label=label
            )

        plt.xlabel("Iterations")
        plt.ylabel("Hypervolume")
        # plt.title("Hypervolume Evolution Across Methods")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        if save_path != "":
            plt.savefig(save_path, bbox_inches="tight")

### MARGIN EVOLUTION
def plot_mu_evolution(
    importance: list[Any], 
    title: str = "Margin Evolution"
) -> None:
    """
    Plots the evolution of mu (e.g., margin or score) over iterations.

    Args:
        importance (list): Values of mu.
        title (str): Plot title.

    Returns:
        None
    """
    plt.figure(figsize=(6, 4))
    plt.plot(importance, color="purple", linewidth=2)
    plt.xlabel("Iterations")
    plt.ylabel("mu")
    plt.title(title)
    plt.grid(True)
    plt.show()


from math import ceil
# TODO: Only for dissertation
### PARETO FRONTIER WITH LIMIT SCALE
def plot_pareto_2d_set_limit_(
    pareto_dict: dict[Any, Any],
    show_subplot: bool = True,
    title: str = "Pareto Frontier (2D)",
    x_label: str = "Objective 1",
    y_label: str = "Objective 2",
    point: npt.NDArray[np.float64] = np.array([]),
    colors: list[str] = ['#d73027', '#fc8d59', '#4575b4', '#fee090', '#91bfdb', '#e0f3f8'],
    axis_mode: str = "global",  # "first" or "global"
) -> None:
    """
    Plots 2D Pareto frontiers with fixed axis limits, useful for comparison.

    Args:
        pareto_dict (dict): Method names mapped to 2D objective arrays.
        show_subplot (bool): Whether to use subplots for each method.
        title (str): Title of the overall figure.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        point (np.ndarray): Optional baseline point.
        colors (list[str]): Color list.
        axis_mode (str): 'first' to use limits from the first method, 'global' to compute global limits.

    Returns:
        None
    """
    n_methods = len(pareto_dict)

    # Define layout: 1 row if <= 3 methods, otherwise 2 rows
    if n_methods <= 3:
        nrows, ncols = 1, n_methods
    else:
        nrows, ncols = 2, ceil(n_methods / 2)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    all_data = np.concatenate(list(pareto_dict.values()), axis=0)
    limits = []
    for i in range(all_data.shape[1]):
        limits.append(
            (
                all_data[:, i].min() - all_data[:, i].std(),
                all_data[:, i].max() + all_data[:, i].std(),
            )
        )
    # fix axis
    if axis_mode == "first":
        first_objs = next(iter(pareto_dict.values()))
        x_min, x_max = first_objs[:, 0].min() - first_objs[:, 0].std(), first_objs[:, 0].max() + first_objs[:, 0].std()
        y_min, y_max = first_objs[:, 1].min() - first_objs[:, 1].std(), first_objs[:, 1].max() + first_objs[:, 1].std()
    elif axis_mode == "global":
        all_points = np.vstack(list(pareto_dict.values()))
        x_min, x_max = all_points[:, 0].min() - all_points[:, 0].std(), all_points[:, 0].max() + all_points[:, 0].std()
        y_min, y_max = all_points[:, 1].min() - all_points[:, 1].std(), all_points[:, 1].max() + all_points[:, 1].std()
    else:
        raise ValueError("axis_mode must be 'first' or 'global'")

    aux = 0
    for i, (method, objs) in enumerate(pareto_dict.items()):
        ax = axes[i]
        if len(point) != 0:
            point = np.asarray(point).flatten()
            ax.scatter(
                x=[point[0]],
                y=[point[1]],
                color="#FFC145",
                s=30,
                label="Baseline",
                edgecolors="black",
                linewidths=0.5,
            )
        ax.scatter(
            x=objs[:, 0],
            y=objs[:, 1],
            label=method,
            s=30,
            color=colors[i % len(colors)],
            edgecolors="black",
            linewidths=0.5,
        )
        ax.set_title(f"{method}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True)

        aux = i

    # Hide extra axes (if there are more subplots than methods)
    for j in range(aux + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()



def plot_pareto_set_limit(
    methods: dict,
    alpha: float = 1.0,
    point: npt.NDArray[np.float64] | list[float] = np.array([]),
    save_path: str  = "",
    labels: tuple | list[str] = (),
    # color       = ['#FF5C8D', '#4BD6A0', '#9D5CFF'],
    color: list[str] | str = ['#d73027', '#fc8d59', '#4575b4', '#fee090', '#91bfdb', '#e0f3f8'],
    point_label: str = "Baseline Model",
    title: str = "Pareto Frontier",
    fontsize: int = 20,
    figsize: tuple[int, int] = (10, 7),
    s_factor: int = 2,
    axis_mode: str = "global",  # "first" or "global"
) -> None:
    num_plots: int = len(methods)
    if isinstance(color, str):
        color = [color]

    layout_configs: dict[int, tuple[int, int]] = {
        1: (1, 1),
        2: (1, 2),
        3: (1, 3),
        4: (2, 2),
        5: (2, 3),
        6: (2, 3),
        7: (3, 3),
        8: (3, 3),
        9: (3, 3),
    }

    nrows, ncols = layout_configs.get(
        num_plots,
        (
            int(np.ceil(np.sqrt(num_plots))),
            int(np.ceil(num_plots / np.ceil(np.sqrt(num_plots)))),
        ),
    )
    fig = plt.figure(figsize=(ncols * figsize[0], nrows * figsize[1]))
    gs = gridspec.GridSpec(nrows, ncols, figure=fig)

    axes = []
    positions = [(i, j) for i in range(nrows) for j in range(ncols)]

    all_data = np.concatenate(list(methods.values()), axis=0)

    num_objectives = all_data.shape[1]
    default_labels = [f"Objective {i+1}" for i in range(num_objectives)]
    labels = labels if len(labels) != 0 else default_labels

    if num_objectives < 4:
        limits = []
        for i in range(all_data.shape[1]):
            #limits.append(
            #    (
            #        all_data[:, i].min() - all_data[:, i].std(),
            #        all_data[:, i].max() + all_data[:, i].std(),
            #    )
            #)
            if axis_mode == "first":
                first_objs = next(iter(methods.values()))
                x_min, x_max = first_objs[:, 0].min() - first_objs[:, 0].std(), first_objs[:, 0].max() + first_objs[:, 0].std()
                y_min, y_max = first_objs[:, 1].min() - first_objs[:, 1].std(), first_objs[:, 1].max() + first_objs[:, 1].std()
                if num_objectives == 3:
                    z_min, z_max = first_objs[:, 2].min() - first_objs[:, 2].std(), first_objs[:, 2].max() + first_objs[:, 2].std()
            elif axis_mode == "global":
                all_points = np.vstack(list(methods.values()))
                x_min, x_max = all_points[:, 0].min() - all_points[:, 0].std(), all_points[:, 0].max() + all_points[:, 0].std()
                y_min, y_max = all_points[:, 1].min() - all_points[:, 1].std(), all_points[:, 1].max() + all_points[:, 1].std()
                if num_objectives == 3:
                    z_min, z_max = all_points[:, 2].min() - all_points[:, 2].std(), all_points[:, 2].max() + all_points[:, 2].std()
            else:
                raise ValueError("axis_mode must be 'first' or 'global'")
            limits.append((x_min, x_max))
            limits.append((y_min, y_max))
            if num_objectives == 3:
                limits.append((z_min, z_max))
        for idx, (key, values) in enumerate(methods.items()):
            row, col = positions[idx]

            ax: Any = None
            if values.shape[1] == 3:
                ax = fig.add_subplot(gs[row, col], projection="3d")
            else:
                ax = fig.add_subplot(gs[row, col])

            axes.append(ax)

            c = color[idx % len(color)]
            ax.grid(True)
            if values.shape[1] == 2:
                ax.scatter(
                    x=values[:, 0],
                    y=values[:, 1],
                    alpha=alpha,
                    c=c,
                    label=f"{key.upper().replace("_", " ")} Solutions",
                    edgecolor="k",
                    linewidth=1.5 * (s_factor // 2),
                    s=50 * s_factor,
                )
            else:
                ax.scatter(
                    xs=values[:, 0],
                    ys=values[:, 1],
                    zs=values[:, 2],
                    alpha=alpha,
                    c=c,
                    label=f"{key.upper().replace("_", " ")} Solutions",
                    edgecolor="k",
                    linewidth=1.5 * (s_factor // 2),
                    s=50 * s_factor,
                )

                ax.set_zlabel(
                    labels[2].replace("_", " ").capitalize(),
                    fontsize=fontsize,
                    labelpad=3 * (figsize[0] // 2),
                )
                ax.tick_params(axis="z", labelsize=fontsize - 5)
                ax.set_zlim(limits[2])

            ax.set_xlabel(
                labels[0].replace("_", " ").capitalize(),
                fontsize=fontsize,
                labelpad=3 * (figsize[0] // 2),
            )
            ax.set_ylabel(
                labels[1].replace("_", " ").capitalize(),
                fontsize=fontsize,
                labelpad=3 * (figsize[0] // 2),
            )
            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])

            ax.legend(fontsize=fontsize, loc="upper center")
            ax.tick_params(axis="x", labelsize=fontsize - 5)
            ax.tick_params(axis="y", labelsize=fontsize - 5)

            if len(point) != 0:
                if values.shape[1] == 2:
                    ax.scatter(
                        x=point[0],
                        y=point[1],
                        marker="X",
                        c="black",
                        # c='#FFC145',
                        edgecolors="k",
                        s=100 * s_factor,
                        linewidth=1.5 * (s_factor // 2),
                        label=point_label,
                    )
                else:
                    ax.scatter(
                        xs=point[0],
                        ys=point[1],
                        zs=point[2],
                        marker="X",
                        c="black",
                        # c='#FFC145',
                        edgecolors="k",
                        s=100 * s_factor,
                        linewidth=1.5 * (s_factor // 2),
                        label=point_label,
                    )

                ax.legend(fontsize=fontsize - 5)

    total_subplots = nrows * ncols
    for idx in range(num_plots, total_subplots):
        fig.add_subplot(gs[positions[idx][0], positions[idx][1]]).axis("off")

    #fig.suptitle(title, fontsize=fontsize + 5)
    fig.tight_layout()  # rect=[0, 0.1, 1, 1])
    # fig.subplots_adjust(wspace=-.3)
    if save_path != "":
        plt.savefig(
            save_path,
            # transparent=True,
            bbox_inches="tight",
        )
    plt.show()

