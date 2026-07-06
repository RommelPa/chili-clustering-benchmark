from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
from sklearn.cluster import (
    AgglomerativeClustering,
    Birch,
    DBSCAN,
    KMeans,
    SpectralClustering,
)
from sklearn.mixture import GaussianMixture


@dataclass(frozen=True)
class ClusteringRunSpec:
    """Single clustering configuration generated from the experiment grid."""

    run_id: str
    algorithm_name: str
    family: str
    role: str
    params: dict[str, Any]


def get_active_algorithms(experiments_config: dict[str, Any]) -> list[str]:
    """Return active algorithms declared in the experiment configuration."""
    return [
        name
        for name, spec in experiments_config['algoritmos'].items()
        if spec.get('activo', False)
    ]


def expand_parameter_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Expand a parameter grid into a list of parameter dictionaries."""
    if not grid:
        return [{}]

    keys = list(grid.keys())
    combinations = product(*(grid[key] for key in keys))

    return [
        dict(zip(keys, values, strict=True))
        for values in combinations
    ]


def build_run_specs(
    experiments_config: dict[str, Any],
) -> list[ClusteringRunSpec]:
    """Build clustering run specifications from the experiment configuration."""
    specs: list[ClusteringRunSpec] = []

    for algorithm_name, algorithm_spec in experiments_config['algoritmos'].items():
        if not algorithm_spec.get('activo', False):
            continue

        parameter_grid = expand_parameter_grid(algorithm_spec['grid'])

        for index, params in enumerate(parameter_grid, start=1):
            specs.append(
                ClusteringRunSpec(
                    run_id=f'{algorithm_name}_{index:03d}',
                    algorithm_name=algorithm_name,
                    family=algorithm_spec['familia'],
                    role=algorithm_spec['rol'],
                    params=params,
                )
            )

    return specs


def fit_predict_clustering(
    spec: ClusteringRunSpec,
    values: np.ndarray,
) -> np.ndarray:
    """Fit a clustering algorithm and return cluster labels."""
    if spec.algorithm_name == 'kmeans':
        model = KMeans(**spec.params)
        labels = model.fit_predict(values)

    elif spec.algorithm_name == 'agglomerative':
        model = AgglomerativeClustering(**spec.params)
        labels = model.fit_predict(values)

    elif spec.algorithm_name == 'gaussian_mixture':
        model = GaussianMixture(**spec.params)
        labels = model.fit_predict(values)

    elif spec.algorithm_name == 'spectral':
        model = SpectralClustering(**spec.params)
        labels = model.fit_predict(values)

    elif spec.algorithm_name == 'birch':
        model = Birch(**spec.params)
        labels = model.fit_predict(values)

    elif spec.algorithm_name == 'dbscan':
        model = DBSCAN(**spec.params)
        labels = model.fit_predict(values)

    else:
        raise ValueError(f'Unsupported clustering algorithm: {spec.algorithm_name}')

    return labels.astype(int)


def count_effective_clusters(labels: np.ndarray) -> int:
    """Count clusters excluding DBSCAN noise label -1."""
    unique_labels = set(labels.tolist())
    unique_labels.discard(-1)

    return len(unique_labels)


def count_noise_points(labels: np.ndarray) -> int:
    """Count observations labeled as noise."""
    return int(np.sum(labels == -1))


def get_cluster_sizes(
    labels: np.ndarray,
    include_noise: bool = False,
) -> dict[int, int]:
    """Return cluster sizes, optionally including noise label -1."""
    unique_labels, counts = np.unique(labels, return_counts=True)

    sizes: dict[int, int] = {}

    for label, count in zip(unique_labels, counts, strict=True):
        label_int = int(label)

        if label_int == -1 and not include_noise:
            continue

        sizes[label_int] = int(count)

    return sizes


def is_valid_partition(
    labels: np.ndarray,
    restrictions: dict[str, Any],
) -> bool:
    """Check whether a partition satisfies configured structural restrictions."""
    n_clusters = count_effective_clusters(labels)

    min_clusters = int(restrictions['min_clusters_validos'])
    max_clusters = int(restrictions['max_clusters_validos'])
    min_cluster_size = int(restrictions['min_tamano_cluster'])

    if n_clusters < min_clusters or n_clusters > max_clusters:
        return False

    cluster_sizes = get_cluster_sizes(labels, include_noise=False)

    if not cluster_sizes:
        return False

    if min(cluster_sizes.values()) < min_cluster_size:
        return False

    noise_points = count_noise_points(labels)

    if noise_points > 0:
        allow_noise = bool(restrictions['permitir_ruido_dbscan'])
        max_noise_ratio = float(restrictions['max_proporcion_ruido'])

        if not allow_noise:
            return False

        if noise_points / len(labels) > max_noise_ratio:
            return False

    return True
