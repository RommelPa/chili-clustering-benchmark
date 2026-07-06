import math

import numpy as np

from src.metrics import (
    build_metrics_row,
    can_compute_internal_metrics,
    evaluate_partition,
    remove_noise_points,
    safe_internal_metrics,
    summarize_partition,
)


def build_values() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [5.0, 5.0],
            [5.1, 4.9],
            [10.0, 10.0],
            [10.1, 9.9],
        ]
    )


def build_restrictions() -> dict[str, int | float | bool]:
    return {
        'min_clusters_validos': 2,
        'max_clusters_validos': 5,
        'min_tamano_cluster': 2,
        'permitir_ruido_dbscan': True,
        'max_proporcion_ruido': 0.25,
    }


def test_remove_noise_points_excludes_minus_one_labels():
    values = build_values()
    labels = np.array([0, 0, 1, 1, -1, -1])

    values_no_noise, labels_no_noise = remove_noise_points(values, labels)

    assert values_no_noise.shape == (4, 2)
    assert labels_no_noise.tolist() == [0, 0, 1, 1]


def test_can_compute_internal_metrics_requires_at_least_two_clusters():
    assert can_compute_internal_metrics(np.array([0, 0, 1, 1])) is True
    assert can_compute_internal_metrics(np.array([0, 0, 0, 0])) is False


def test_safe_internal_metrics_returns_values_for_valid_partition():
    values = build_values()
    labels = np.array([0, 0, 1, 1, 2, 2])

    metrics = safe_internal_metrics(values, labels)

    assert not math.isnan(metrics['silhouette'])
    assert not math.isnan(metrics['davies_bouldin'])
    assert not math.isnan(metrics['calinski_harabasz'])
    assert metrics['davies_bouldin'] >= 0


def test_safe_internal_metrics_returns_nan_for_invalid_partition():
    values = build_values()
    labels = np.array([0, 0, 0, 0, 0, 0])

    metrics = safe_internal_metrics(values, labels)

    assert math.isnan(metrics['silhouette'])
    assert math.isnan(metrics['davies_bouldin'])
    assert math.isnan(metrics['calinski_harabasz'])


def test_summarize_partition_counts_noise_and_cluster_sizes():
    labels = np.array([0, 0, 1, 1, -1])

    summary = summarize_partition(labels)

    assert summary['n_clusters'] == 2
    assert summary['n_noise'] == 1
    assert summary['noise_ratio'] == 0.2
    assert summary['cluster_min_size'] == 2
    assert summary['cluster_max_size'] == 2


def test_evaluate_partition_marks_valid_partition():
    values = build_values()
    labels = np.array([0, 0, 1, 1, 2, 2])

    result = evaluate_partition(values, labels, build_restrictions())

    assert result['is_valid_partition'] is True
    assert result['n_clusters'] == 3
    assert result['cluster_min_size'] == 2


def test_evaluate_partition_marks_invalid_small_cluster():
    values = build_values()
    labels = np.array([0, 0, 0, 0, 1, -1])

    result = evaluate_partition(values, labels, build_restrictions())

    assert result['is_valid_partition'] is False
    assert result['cluster_min_size'] == 1


def test_build_metrics_row_flattens_metadata_and_params():
    values = build_values()
    labels = np.array([0, 0, 1, 1, 2, 2])

    row = build_metrics_row(
        run_id='kmeans_001',
        algorithm_name='kmeans',
        family='particional_centroides',
        role='candidato_principal',
        feature_set_name='base_operativa',
        scaler_name='zscore',
        params={'n_clusters': 3, 'random_state': 42},
        values=values,
        labels=labels,
        restrictions=build_restrictions(),
    )

    assert row['run_id'] == 'kmeans_001'
    assert row['algorithm_name'] == 'kmeans'
    assert row['feature_set_name'] == 'base_operativa'
    assert row['param_n_clusters'] == 3
    assert row['param_random_state'] == 42
    assert row['is_valid_partition'] is True
