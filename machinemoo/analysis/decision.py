import numpy as np
from typing import Any

def get_best_median_solution(
    objectives: list[np.array],
    models: list[Any]
) -> tuple[list[float], Any, float]:
    """
    Retorna a solução (vetor de objetivos) com a menor mediana,
    o modelo correspondente e o valor da mediana.

    Args:
        objectives: Lista de listas, onde cada sublista contém os valores de objetivos de uma solução.
        models: Lista de modelos ou objetos associados, um para cada solução.

    Returns:
        Uma tupla com:
            - A solução (lista de objetivos) com menor mediana,
            - O modelo correspondente,
            - O valor da mediana.
    """
    if len(objectives) != len(models):
        raise ValueError("Length of objectives and models must be the same.")

    # Filtra soluções que não contêm 0.0
    filtered = [
        (i, obj, model) for i, (obj, model) in enumerate(zip(objectives, models))
        if not np.any(np.equal(obj, 0.0))
    ]

    if not filtered:
        raise ValueError("No valid solutions remaining after filtering out 0.0 objective values.")
    
    # Separa os componentes filtrados
    indices, filtered_objs, filtered_models = zip(*filtered)

    medians = [np.median(obj) for obj in filtered_objs]
    best_idx_in_filtered = int(np.argmin(medians))
    original_idx = indices[best_idx_in_filtered]

    return (
        filtered_objs[best_idx_in_filtered],
        filtered_models[best_idx_in_filtered],
        medians[best_idx_in_filtered],
        original_idx  # índice em relação à lista original
    )

def get_best_weighted_solution(
    objectives: list[np.array],
    models: list[Any],
    weights: list[float]
) -> tuple[list[float], Any, float]:
    """
    Retorna a solução com a menor média ponderada dos objetivos, dado um vetor de pesos.

    Args:
        objectives: Lista de listas, onde cada sublista contém os valores de objetivos de uma solução.
        models: Lista de modelos ou objetos associados, um para cada solução.
        weights: Vetor de pesos (deve ter o mesmo tamanho que o número de objetivos por solução).

    Returns:
        Uma tupla com:
            - A solução (lista de objetivos) com menor média ponderada,
            - O modelo correspondente,
            - O valor da média ponderada.
    """
    
    if len(objectives) != len(models):
        raise ValueError("Length of objectives and models must be the same.")
    
    num_objectives = len(objectives[0])
    if any(len(obj) != num_objectives for obj in objectives):
        raise ValueError("All objective vectors must have the same length.")
    
    if len(weights) != num_objectives:
        raise ValueError("Weights vector must have the same length as each objective vector.")

    # Filtra soluções que não contêm 0.0
    filtered = [
        (i, obj, model) for i, (obj, model) in enumerate(zip(objectives, models))
        if not np.any(np.isclose(obj, 0.0))
    ]

    if not filtered:
        raise ValueError("No valid solutions remaining after filtering out 0.0 objective values.")

    # Desempacota os dados filtrados
    indices, filtered_objs, filtered_models = zip(*filtered)

    weights = np.array(weights)
    weights = weights / weights.sum()  # normaliza para somar 1

    weighted_means = [np.dot(obj, weights) for obj in filtered_objs]
    best_idx_in_filtered = int(np.argmin(weighted_means))
    original_idx = indices[best_idx_in_filtered]

    return (
        filtered_objs[best_idx_in_filtered],
        filtered_models[best_idx_in_filtered],
        weighted_means[best_idx_in_filtered],
        original_idx  # índice em relação à lista original
    )