from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from src.clustering_models import (
    count_effective_clusters,
    count_noise_points,
    get_cluster_sizes,
    is_valid_partition,
)


def remove_noise_points(
    values: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove observations labeled as DBSCAN noise."""
    mask = labels != -1

    return values[mask], labels[mask]


def can_compute_internal_metrics(labels: np.ndarray) -> bool:
    """Check whether internal clustering metrics can be computed."""
    n_samples = len(labels)
    n_clusters = count_effective_clusters(labels)

    return 2 <= n_clusters <= n_samples - 1


def safe_internal_metrics(
    values: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Compute internal metrics safely, returning NaN when not applicable."""
    values_no_noise, labels_no_noise = remove_noise_points(values, labels)

    if not can_compute_internal_metrics(labels_no_noise):
        return {
            'silhouette': float('nan'),
            'davies_bouldin': float('nan'),
            'calinski_harabasz': float('nan'),
        }

    return {
        'silhouette': float(silhouette_score(values_no_noise, labels_no_noise)),
        'davies_bouldin': float(
            davies_bouldin_score(values_no_noise, labels_no_noise)
        ),
        'calinski_harabasz': float(
            calinski_harabasz_score(values_no_noise, labels_no_noise)
        ),
    }


def summarize_partition(labels: np.ndarray) -> dict[str, int | float]:
    """Summarize structural properties of a clustering partition."""
    cluster_sizes = get_cluster_sizes(labels, include_noise=False)

    if cluster_sizes:
        cluster_min_size = min(cluster_sizes.values())
        cluster_max_size = max(cluster_sizes.values())
    else:
        cluster_min_size = 0
        cluster_max_size = 0

    n_noise = count_noise_points(labels)

    return {
        'n_clusters': count_effective_clusters(labels),
        'n_noise': n_noise,
        'noise_ratio': n_noise / len(labels),
        'cluster_min_size': cluster_min_size,
        'cluster_max_size': cluster_max_size,
    }


def evaluate_partition(
    values: np.ndarray,
    labels: np.ndarray,
    restrictions: dict[str, Any],
) -> dict[str, int | float | bool]:
    """Evaluate a clustering partition using structural and internal metrics."""
    structural = summarize_partition(labels)
    internal = safe_internal_metrics(values, labels)

    return {
        **structural,
        **internal,
        'is_valid_partition': is_valid_partition(labels, restrictions),
    }


def build_metrics_row(
    run_id: str,
    algorithm_name: str,
    family: str,
    role: str,
    feature_set_name: str,
    scaler_name: str,
    params: dict[str, Any],
    values: np.ndarray,
    labels: np.ndarray,
    restrictions: dict[str, Any],
) -> dict[str, Any]:
    """Build a flat metrics row for benchmark reporting."""
    metrics = evaluate_partition(values, labels, restrictions)

    row: dict[str, Any] = {
        'run_id': run_id,
        'algorithm_name': algorithm_name,
        'family': family,
        'role': role,
        'feature_set_name': feature_set_name,
        'scaler_name': scaler_name,
        **metrics,
    }

    for param_name, param_value in params.items():
        row[f'param_{param_name}'] = param_value

    return row
