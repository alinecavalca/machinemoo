import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

def plot_pareto_2d(
    solutions,
    title = "2 Dimension Pareto Frontier",
    labels = ["Objective 1", "Objective 2"],
    legend = "Pareto solutions",
    point = None,
    point_label = "Baseline Model",
    save_path = None,
    color = ["#FF5C8D"],
    alpha = 1.0
) -> None:
    pass


def plot_pareto_3d(
    solutions,
    title = "3 Dimension Pareto Frontier",
    labels = ["Objective 1", "Objective 2",  "Objective 3"],
    legend = "Pareto solutions",
    point = None,
    point_label = "Baseline Model",
    save_path = None,
    color = ["#FF5C8D"],
    alpha = 1.0
) -> None:
    pass

def plot_parallel_coordinates(
    solutions,
    title = "Parallel Coordinates",
    labels = None,
    color = ["#FF5C8D"],
    colormap: str = "viridis",
    point = None,
    point_label = "Baseline Model",
    point_color = "blue",
    alpha = 1.0
) -> None:

    num_objectives = len(solutions[0])
    default_labels = [f"Objective {i+1}" for i in range(num_objectives)]
    columns = labels if labels else default_labels

    df = pd.DataFrame(solutions, columns=columns)
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

    if point is not None:
        point = np.array(point).reshape(1, -1)
        df_point = pd.DataFrame(point, columns=columns)
        df_point["Solution"] = [point_label]

        parallel_coordinates(
            df_point,
            class_column="Solution",
            color=point_color,
            alpha=alpha
        )

    plt.title(title, fontsize=16)
    plt.xlabel("Objectives", fontsize=14)
    plt.ylabel("Values", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.show()


def plot_hypervolume(
    hypervolume_values,
    method = "",
    color = "tab:pink"
) -> None:
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
    methods_hv,
    subplots = False,
    colors = None,
) -> None:
    pass

def plot_mu_evolution(
        mu_values,
        title: str = "Margin Evolution"
) -> None:

    plt.figure(figsize=(6, 4))
    plt.plot(mu_values, color="purple", linewidth=2)
    plt.xlabel("Iterations")
    plt.ylabel("mu")
    plt.title(title)
    plt.grid(True)
    plt.show()