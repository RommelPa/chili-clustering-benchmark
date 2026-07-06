import pandas as pd
import pytest

from src.ranking import (
    add_normalized_metrics,
    add_penalty_columns,
    filter_rankable_runs,
    min_max_scale,
    rank_benchmark_runs,
    save_ranking_outputs,
    summarize_top_by_algorithm,
)


def build_config(tmp_path=None) -> dict:
    benchmark_path = 'outputs/tables/benchmark_runs.csv'

    if tmp_path is not None:
        benchmark_path = str(tmp_path / 'benchmark_runs.csv')

    return {
        'ranking': {
            'pesos': {
                'silhouette': 0.25,
                'davies_bouldin': 0.20,
                'calinski_harabasz': 0.20,
                'estabilidad': 0.25,
                'penalizacion_ruido': 0.05,
                'penalizacion_cluster_pequeno': 0.05,
            }
        },
        'salidas': {
            'tablas': {
                'benchmark_runs': benchmark_path,
            }
        },
    }


def build_benchmark() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                'run_id': 'kmeans_001',
                'algorithm_name': 'kmeans',
                'family': 'particional',
                'role': 'principal',
                'feature_set_name': 'base_operativa',
                'scaler_name': 'zscore',
                'status': 'success',
                'is_valid_partition': True,
                'n_clusters': 3,
                'n_noise': 0,
                'noise_ratio': 0.0,
                'cluster_min_size': 3,
                'cluster_max_size': 6,
                'silhouette': 0.50,
                'davies_bouldin': 0.60,
                'calinski_harabasz': 30.0,
            },
            {
                'run_id': 'gmm_001',
                'algorithm_name': 'gaussian_mixture',
                'family': 'probabilistico',
                'role': 'principal',
                'feature_set_name': 'base_operativa',
                'scaler_name': 'zscore',
                'status': 'success',
                'is_valid_partition': True,
                'n_clusters': 3,
                'n_noise': 0,
                'noise_ratio': 0.0,
                'cluster_min_size': 4,
                'cluster_max_size': 5,
                'silhouette': 0.60,
                'davies_bouldin': 0.50,
                'calinski_harabasz': 40.0,
            },
            {
                'run_id': 'dbscan_001',
                'algorithm_name': 'dbscan',
                'family': 'densidad',
                'role': 'exploratorio',
                'feature_set_name': 'base_operativa',
                'scaler_name': 'zscore',
                'status': 'success',
                'is_valid_partition': True,
                'n_clusters': 2,
                'n_noise': 1,
                'noise_ratio': 0.10,
                'cluster_min_size': 2,
                'cluster_max_size': 7,
                'silhouette': 0.55,
                'davies_bouldin': 0.70,
                'calinski_harabasz': 20.0,
            },
            {
                'run_id': 'bad_001',
                'algorithm_name': 'kmeans',
                'family': 'particional',
                'role': 'principal',
                'feature_set_name': 'base_operativa',
                'scaler_name': 'zscore',
                'status': 'failed',
                'is_valid_partition': False,
                'n_clusters': 0,
                'n_noise': 0,
                'noise_ratio': 0.0,
                'cluster_min_size': 0,
                'cluster_max_size': 0,
                'silhouette': None,
                'davies_bouldin': None,
                'calinski_harabasz': None,
            },
        ]
    )


def test_min_max_scale_higher_is_better():
    values = pd.Series([10, 20, 30])

    scaled = min_max_scale(values, higher_is_better=True)

    assert scaled.tolist() == [0.0, 0.5, 1.0]


def test_min_max_scale_lower_is_better():
    values = pd.Series([10, 20, 30])

    scaled = min_max_scale(values, higher_is_better=False)

    assert scaled.tolist() == [1.0, 0.5, 0.0]


def test_filter_rankable_runs_keeps_successful_valid_complete_runs():
    benchmark = build_benchmark()

    rankable = filter_rankable_runs(benchmark)

    assert rankable['run_id'].tolist() == [
        'kmeans_001',
        'gmm_001',
        'dbscan_001',
    ]


def test_filter_rankable_runs_raises_when_no_rankable_runs():
    benchmark = build_benchmark()
    benchmark['status'] = 'failed'

    with pytest.raises(ValueError, match='No rankable benchmark runs'):
        filter_rankable_runs(benchmark)


def test_add_normalized_metrics_creates_score_inputs():
    benchmark = filter_rankable_runs(build_benchmark())

    ranked = add_normalized_metrics(benchmark)

    assert 'silhouette_norm' in ranked.columns
    assert 'davies_bouldin_norm' in ranked.columns
    assert 'calinski_harabasz_norm' in ranked.columns


def test_add_penalty_columns_calculates_cluster_imbalance():
    benchmark = filter_rankable_runs(build_benchmark())

    ranked = add_penalty_columns(benchmark)

    assert 'cluster_imbalance' in ranked.columns
    assert ranked.loc[ranked['run_id'] == 'gmm_001', 'cluster_imbalance'].iloc[0] == pytest.approx(0.2)


def test_rank_benchmark_runs_orders_best_configuration_first():
    benchmark = build_benchmark()
    config = build_config()

    ranked = rank_benchmark_runs(benchmark, config)

    assert ranked.iloc[0]['rank'] == 1
    assert ranked.iloc[0]['run_id'] == 'gmm_001'
    assert ranked['ranking_score'].is_monotonic_decreasing


def test_summarize_top_by_algorithm_keeps_one_per_algorithm():
    benchmark = build_benchmark()
    config = build_config()
    ranked = rank_benchmark_runs(benchmark, config)

    top = summarize_top_by_algorithm(ranked)

    assert top['algorithm_name'].nunique() == len(top)
    assert set(top['algorithm_name']) == {
        'kmeans',
        'gaussian_mixture',
        'dbscan',
    }


def test_save_ranking_outputs_writes_csv_files(tmp_path):
    benchmark = build_benchmark()
    config = build_config(tmp_path)
    ranked = rank_benchmark_runs(benchmark, config)
    top = summarize_top_by_algorithm(ranked)

    ranking_path, top_path = save_ranking_outputs(ranked, top, config)

    assert ranking_path.exists()
    assert top_path.exists()

