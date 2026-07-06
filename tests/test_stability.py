import numpy as np
import pandas as pd
import pytest

from src.clustering_models import ClusteringRunSpec
from src.experiment_matrix import ExperimentMatrix
from src.stability import (
    build_matrix_map,
    build_spec_map,
    calculate_pairwise_stability,
    compute_run_stability,
    get_project_random_state,
    get_sample_size,
    save_stability_outputs,
)


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
    frame = pd.DataFrame(
        {
            'AH': [f'AH-{2012 + index}' for index in range(6)],
            'AH_YEAR': [2012 + index for index in range(6)],
            'x1': values[:, 0],
            'x2': values[:, 1],
        }
    )

    return ExperimentMatrix(
        feature_set_name='base_operativa',
        scaler_name='zscore',
        variables=['x1', 'x2'],
        ah_labels=frame['AH'].tolist(),
        ah_years=frame['AH_YEAR'].tolist(),
        values=values,
        scaled_frame=frame,
    )


def test_get_sample_size_uses_fraction_with_bounds():
    assert get_sample_size(14, 0.85) == 12
    assert get_sample_size(1, 0.85) == 1
    assert get_sample_size(14, 2.0) == 14


def test_calculate_pairwise_stability_is_label_invariant():
    reference = np.array([0, 0, 1, 1])
    candidate = np.array([5, 5, 2, 2])

    result = calculate_pairwise_stability(reference, candidate)

    assert result['stability_score'] == pytest.approx(1.0)
    assert result['stability_intra'] == pytest.approx(1.0)
    assert result['stability_inter'] == pytest.approx(1.0)


def test_calculate_pairwise_stability_detects_membership_change():
    reference = np.array([0, 0, 1, 1])
    candidate = np.array([0, 1, 0, 1])

    result = calculate_pairwise_stability(reference, candidate)

    assert result['stability_score'] < 1.0


def test_calculate_pairwise_stability_raises_for_different_lengths():
    with pytest.raises(ValueError, match='same length'):
        calculate_pairwise_stability(
            np.array([0, 0, 1]),
            np.array([0, 0]),
        )


def test_build_matrix_map_uses_feature_set_and_scaler_key():
    matrix = build_matrix()

    matrix_map = build_matrix_map([matrix])

    assert ('base_operativa', 'zscore') in matrix_map


def test_compute_run_stability_returns_summary_and_iterations():
    matrix = build_matrix()
    spec = ClusteringRunSpec(
        run_id='kmeans_001',
        algorithm_name='kmeans',
        family='particional',
        role='test',
        params={
            'n_clusters': 3,
            'n_init': 10,
            'max_iter': 100,
            'random_state': 42,
        },
    )
    reference_labels = np.array([0, 0, 1, 1, 2, 2])

    summary, iterations = compute_run_stability(
        spec=spec,
        matrix=matrix,
        reference_labels=reference_labels,
        n_iterations=5,
        sample_fraction=0.85,
        random_state=42,
    )

    assert summary['n_iterations'] == 5
    assert len(iterations) == 5
    assert 'stability_score' in iterations.columns


def test_build_spec_map_contains_configured_run_ids():
    config = {
        'algoritmos': {
            'kmeans': {
                'activo': True,
                'familia': 'particional',
                'rol': 'test',
                'grid': {
                    'n_clusters': [2],
                    'n_init': [10],
                },
            }
        }
    }

    spec_map = build_spec_map(config)

    assert 'kmeans_001' in spec_map


def test_save_stability_outputs_writes_csv_files(tmp_path):
    config = {
        'salidas': {
            'tablas': {
                'benchmark_runs': str(tmp_path / 'benchmark_runs.csv')
            }
        }
    }
    rows = [
        {
            'run_id': 'kmeans_001',
            'stability_score': 1.0,
        }
    ]
    iterations = pd.DataFrame(
        {
            'run_id': ['kmeans_001'],
            'iteration': [1],
            'stability_score': [1.0],
        }
    )

    stability_path, iterations_path = save_stability_outputs(
        stability_rows=rows,
        iteration_frames=[iterations],
        experiments_config=config,
    )

    assert stability_path.exists()
    assert iterations_path.exists()


def test_get_project_random_state_accepts_seed_key():
    config = {
        'proyecto': {
            'seed': 42,
        }
    }

    assert get_project_random_state(config) == 42


def test_get_project_random_state_prefers_random_state_key():
    config = {
        'proyecto': {
            'seed': 42,
            'random_state': 123,
        }
    }

    assert get_project_random_state(config) == 123
