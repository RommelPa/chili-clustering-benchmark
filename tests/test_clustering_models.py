import numpy as np
import pytest

from src.clustering_models import (
    ClusteringRunSpec,
    build_run_specs,
    count_effective_clusters,
    count_noise_points,
    expand_parameter_grid,
    fit_predict_clustering,
    get_active_algorithms,
    get_cluster_sizes,
    is_valid_partition,
)
from src.config_loader import load_experiments_config


def build_synthetic_matrix() -> np.ndarray:
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


def test_get_active_algorithms_returns_six_algorithms():
    config = load_experiments_config()

    algorithms = get_active_algorithms(config)

    assert algorithms == [
        'kmeans',
        'agglomerative',
        'gaussian_mixture',
        'spectral',
        'birch',
        'dbscan',
    ]


def test_expand_parameter_grid_returns_cartesian_product():
    grid = {
        'n_clusters': [2, 3],
        'n_init': [50, 100],
    }

    expanded = expand_parameter_grid(grid)

    assert len(expanded) == 4
    assert {'n_clusters': 2, 'n_init': 50} in expanded
    assert {'n_clusters': 3, 'n_init': 100} in expanded


def test_build_run_specs_returns_expected_number_of_configurations():
    config = load_experiments_config()

    specs = build_run_specs(config)

    assert len(specs) == 70
    assert specs[0].algorithm_name == 'kmeans'
    assert specs[0].run_id == 'kmeans_001'


def test_fit_predict_kmeans_returns_two_clusters():
    values = build_synthetic_matrix()
    spec = ClusteringRunSpec(
        run_id='kmeans_test',
        algorithm_name='kmeans',
        family='particional_centroides',
        role='test',
        params={
            'n_clusters': 3,
            'n_init': 10,
            'max_iter': 100,
            'random_state': 42,
        },
    )

    labels = fit_predict_clustering(spec, values)

    assert len(labels) == len(values)
    assert count_effective_clusters(labels) == 3


def test_fit_predict_gaussian_mixture_returns_two_clusters():
    values = build_synthetic_matrix()
    spec = ClusteringRunSpec(
        run_id='gmm_test',
        algorithm_name='gaussian_mixture',
        family='probabilistico',
        role='test',
        params={
            'n_components': 3,
            'covariance_type': 'diag',
            'reg_covar': 0.000001,
            'random_state': 42,
            'n_init': 5,
        },
    )

    labels = fit_predict_clustering(spec, values)

    assert len(labels) == len(values)
    assert count_effective_clusters(labels) == 3


def test_fit_predict_raises_for_unknown_algorithm():
    values = build_synthetic_matrix()
    spec = ClusteringRunSpec(
        run_id='unknown_test',
        algorithm_name='unknown',
        family='unknown',
        role='test',
        params={},
    )

    with pytest.raises(ValueError, match='Unsupported clustering algorithm'):
        fit_predict_clustering(spec, values)


def test_count_effective_clusters_excludes_noise():
    labels = np.array([0, 0, 1, 1, -1])

    assert count_effective_clusters(labels) == 2
    assert count_noise_points(labels) == 1


def test_get_cluster_sizes_excludes_noise_by_default():
    labels = np.array([0, 0, 1, 1, -1])

    assert get_cluster_sizes(labels) == {0: 2, 1: 2}
    assert get_cluster_sizes(labels, include_noise=True) == {-1: 1, 0: 2, 1: 2}


def test_is_valid_partition_accepts_balanced_partition_with_limited_noise():
    restrictions = {
        'min_clusters_validos': 2,
        'max_clusters_validos': 5,
        'min_tamano_cluster': 2,
        'permitir_ruido_dbscan': True,
        'max_proporcion_ruido': 0.25,
    }
    labels = np.array([0, 0, 1, 1, -1])

    assert is_valid_partition(labels, restrictions) is True


def test_is_valid_partition_rejects_small_cluster():
    restrictions = {
        'min_clusters_validos': 2,
        'max_clusters_validos': 5,
        'min_tamano_cluster': 2,
        'permitir_ruido_dbscan': True,
        'max_proporcion_ruido': 0.25,
    }
    labels = np.array([0, 0, 0, 1, -1])

    assert is_valid_partition(labels, restrictions) is False
