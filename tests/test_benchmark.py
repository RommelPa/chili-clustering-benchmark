import numpy as np
import pandas as pd

from src.benchmark import (
    build_failed_metrics_row,
    build_labels_frame,
    is_loggable_number,
    run_single_benchmark,
    save_benchmark_outputs,
)
from src.clustering_models import ClusteringRunSpec
from src.experiment_matrix import ExperimentMatrix


def build_matrix() -> ExperimentMatrix:
    values = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [5.0, 5.0],
            [5.1, 4.9],
            [10.0, 10.0],
            [10.1, 9.9],
        ]
    )
    scaled_frame = pd.DataFrame(
        {
            'AH': [
                'AH-2012',
                'AH-2013',
                'AH-2014',
                'AH-2015',
                'AH-2016',
                'AH-2017',
            ],
            'AH_YEAR': [2012, 2013, 2014, 2015, 2016, 2017],
            'x1': values[:, 0],
            'x2': values[:, 1],
        }
    )

    return ExperimentMatrix(
        feature_set_name='base_operativa',
        scaler_name='zscore',
        variables=['x1', 'x2'],
        ah_labels=scaled_frame['AH'].tolist(),
        ah_years=scaled_frame['AH_YEAR'].tolist(),
        values=values,
        scaled_frame=scaled_frame,
    )


def build_restrictions() -> dict[str, int | float | bool]:
    return {
        'min_clusters_validos': 2,
        'max_clusters_validos': 5,
        'min_tamano_cluster': 2,
        'permitir_ruido_dbscan': True,
        'max_proporcion_ruido': 0.25,
    }


def test_is_loggable_number_accepts_only_finite_numbers():
    assert is_loggable_number(1) is True
    assert is_loggable_number(1.5) is True
    assert is_loggable_number(float('nan')) is False
    assert is_loggable_number(True) is False
    assert is_loggable_number('x') is False


def test_build_failed_metrics_row_contains_error_status():
    matrix = build_matrix()
    spec = ClusteringRunSpec(
        run_id='bad_001',
        algorithm_name='bad',
        family='test',
        role='test',
        params={'x': 1},
    )

    row = build_failed_metrics_row(spec, matrix, RuntimeError('boom'))

    assert row['status'] == 'failed'
    assert row['is_valid_partition'] is False
    assert row['error_message'] == 'boom'
    assert row['param_x'] == 1


def test_build_labels_frame_has_one_row_per_year():
    matrix = build_matrix()
    spec = ClusteringRunSpec(
        run_id='kmeans_001',
        algorithm_name='kmeans',
        family='test',
        role='test',
        params={},
    )
    labels = np.array([0, 0, 1, 1, 2, 2])

    labels_frame = build_labels_frame(spec, matrix, labels)

    assert labels_frame.shape == (6, 7)
    assert labels_frame['run_id'].nunique() == 1
    assert labels_frame['cluster_label'].tolist() == [0, 0, 1, 1, 2, 2]


def test_run_single_benchmark_successful_kmeans():
    matrix = build_matrix()
    spec = ClusteringRunSpec(
        run_id='kmeans_001',
        algorithm_name='kmeans',
        family='particional_centroides',
        role='candidato_principal',
        params={
            'n_clusters': 3,
            'n_init': 10,
            'max_iter': 100,
            'random_state': 42,
        },
    )

    row, labels_frame = run_single_benchmark(
        spec=spec,
        matrix=matrix,
        restrictions=build_restrictions(),
    )

    assert row['status'] == 'success'
    assert row['algorithm_name'] == 'kmeans'
    assert row['n_clusters'] == 3
    assert row['is_valid_partition'] is True
    assert labels_frame is not None


def test_run_single_benchmark_captures_failure():
    matrix = build_matrix()
    spec = ClusteringRunSpec(
        run_id='unknown_001',
        algorithm_name='unknown',
        family='unknown',
        role='test',
        params={},
    )

    row, labels_frame = run_single_benchmark(
        spec=spec,
        matrix=matrix,
        restrictions=build_restrictions(),
    )

    assert row['status'] == 'failed'
    assert row['is_valid_partition'] is False
    assert labels_frame is None


def test_save_benchmark_outputs_writes_csv_files(tmp_path):
    rows = [
        {
            'run_id': 'kmeans_001',
            'algorithm_name': 'kmeans',
            'is_valid_partition': True,
            'status': 'success',
        }
    ]
    labels_frame = pd.DataFrame(
        {
            'run_id': ['kmeans_001'],
            'AH': ['AH-2012'],
            'cluster_label': [0],
        }
    )
    config = {
        'salidas': {
            'tablas': {
                'benchmark_runs': str(tmp_path / 'benchmark_runs.csv')
            }
        }
    }

    benchmark_path, labels_path = save_benchmark_outputs(
        rows=rows,
        labels_frames=[labels_frame],
        experiments_config=config,
    )

    assert benchmark_path.exists()
    assert labels_path.exists()
