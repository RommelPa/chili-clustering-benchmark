import pandas as pd

from src.algorithm_analysis import (
    build_algorithm_performance_summary,
    save_algorithm_analysis_outputs,
    select_best_run_by_algorithm,
    summarize_algorithm_configurations,
)


def build_benchmark() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'run_id': [
                'kmeans_001',
                'kmeans_002',
                'spectral_001',
                'dbscan_001',
            ],
            'algorithm_name': ['kmeans', 'kmeans', 'spectral', 'dbscan'],
            'family': [
                'particional_centroides',
                'particional_centroides',
                'afinidad_grafo',
                'densidad',
            ],
            'role': [
                'candidato_principal',
                'candidato_principal',
                'candidato_principal',
                'contraste',
            ],
            'status': ['success', 'success', 'success', 'success'],
            'is_valid_partition': [True, False, True, False],
        }
    )


def build_final_runs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'final_rank': [1, 2],
            'run_id': ['spectral_001', 'kmeans_001'],
            'algorithm_name': ['spectral', 'kmeans'],
            'family': ['afinidad_grafo', 'particional_centroides'],
            'role': ['candidato_principal', 'candidato_principal'],
            'partition_id': ['partition_001', 'partition_001'],
            'final_score': [0.97, 0.96],
            'ranking_score': [0.97, 0.97],
            'stability_score': [0.99, 0.97],
            'n_clusters': [2, 2],
            'silhouette': [0.44, 0.44],
            'davies_bouldin': [0.75, 0.75],
            'calinski_harabasz': [12.5, 12.5],
        }
    )


def test_summarize_algorithm_configurations_counts_valid_runs():
    summary = summarize_algorithm_configurations(build_benchmark())

    kmeans = summary[summary['algorithm_name'] == 'kmeans'].iloc[0]

    assert kmeans['n_configurations'] == 2
    assert kmeans['n_successful_runs'] == 2
    assert kmeans['n_valid_partitions'] == 1
    assert kmeans['valid_partition_rate'] == 0.5


def test_select_best_run_by_algorithm_keeps_one_run_per_algorithm():
    best_runs = select_best_run_by_algorithm(build_final_runs())

    assert set(best_runs['algorithm_name']) == {'kmeans', 'spectral'}
    assert best_runs.iloc[0]['run_id'] == 'spectral_001'


def test_build_algorithm_performance_summary_adds_best_run_information():
    summary = build_algorithm_performance_summary(
        benchmark=build_benchmark(),
        final_runs=build_final_runs(),
    )

    spectral = summary[summary['algorithm_name'] == 'spectral'].iloc[0]

    assert spectral['best_run_id'] == 'spectral_001'
    assert spectral['best_partition_id'] == 'partition_001'
    assert spectral['best_final_score'] == 0.97


def test_build_algorithm_performance_summary_keeps_algorithm_without_final_run():
    summary = build_algorithm_performance_summary(
        benchmark=build_benchmark(),
        final_runs=build_final_runs(),
    )

    dbscan = summary[summary['algorithm_name'] == 'dbscan'].iloc[0]

    assert dbscan['n_configurations'] == 1
    assert dbscan['n_final_ranked_runs'] == 0
    assert pd.isna(dbscan['best_run_id'])


def test_save_algorithm_analysis_outputs_writes_csv_files(tmp_path):
    summary = build_algorithm_performance_summary(
        benchmark=build_benchmark(),
        final_runs=build_final_runs(),
    )
    best_runs = select_best_run_by_algorithm(build_final_runs())
    config = {
        'salidas': {
            'tablas': {
                'benchmark_runs': str(tmp_path / 'benchmark_runs.csv')
            }
        }
    }

    summary_path, best_runs_path = save_algorithm_analysis_outputs(
        algorithm_summary=summary,
        best_runs=best_runs,
        experiments_config=config,
    )

    assert summary_path.exists()
    assert best_runs_path.exists()
