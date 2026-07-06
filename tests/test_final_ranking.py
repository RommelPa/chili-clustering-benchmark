import pandas as pd
import pytest

from src.final_ranking import (
    add_partition_information,
    build_final_partition_ranking,
    calculate_final_scores,
    merge_ranking_and_stability,
    rank_final_runs,
    save_final_outputs,
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


def build_ranking() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                'rank': 1,
                'run_id': 'kmeans_001',
                'algorithm_name': 'kmeans',
                'ranking_score': 0.90,
                'n_clusters': 2,
                'n_noise': 0,
                'cluster_min_size': 4,
                'cluster_max_size': 10,
                'silhouette': 0.44,
                'davies_bouldin': 0.75,
                'calinski_harabasz': 12.0,
                'silhouette_norm': 1.0,
                'davies_bouldin_norm': 1.0,
                'calinski_harabasz_norm': 1.0,
                'noise_penalty_value': 0.0,
                'cluster_size_penalty_value': 0.6,
            },
            {
                'rank': 2,
                'run_id': 'spectral_001',
                'algorithm_name': 'spectral',
                'ranking_score': 0.88,
                'n_clusters': 2,
                'n_noise': 0,
                'cluster_min_size': 4,
                'cluster_max_size': 10,
                'silhouette': 0.44,
                'davies_bouldin': 0.75,
                'calinski_harabasz': 12.0,
                'silhouette_norm': 1.0,
                'davies_bouldin_norm': 1.0,
                'calinski_harabasz_norm': 1.0,
                'noise_penalty_value': 0.0,
                'cluster_size_penalty_value': 0.6,
            },
            {
                'rank': 3,
                'run_id': 'gmm_001',
                'algorithm_name': 'gaussian_mixture',
                'ranking_score': 0.70,
                'n_clusters': 3,
                'n_noise': 0,
                'cluster_min_size': 2,
                'cluster_max_size': 8,
                'silhouette': 0.30,
                'davies_bouldin': 0.90,
                'calinski_harabasz': 8.0,
                'silhouette_norm': 0.4,
                'davies_bouldin_norm': 0.5,
                'calinski_harabasz_norm': 0.3,
                'noise_penalty_value': 0.0,
                'cluster_size_penalty_value': 0.75,
            },
        ]
    )


def build_stability() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                'run_id': 'kmeans_001',
                'stability_score': 0.80,
                'stability_intra': 0.81,
                'stability_inter': 0.79,
                'stability_gap': 0.02,
                'n_successful_iterations': 500,
            },
            {
                'run_id': 'spectral_001',
                'stability_score': 0.99,
                'stability_intra': 0.99,
                'stability_inter': 0.99,
                'stability_gap': 0.00,
                'n_successful_iterations': 500,
            },
            {
                'run_id': 'gmm_001',
                'stability_score': 0.70,
                'stability_intra': 0.72,
                'stability_inter': 0.68,
                'stability_gap': 0.04,
                'n_successful_iterations': 500,
            },
        ]
    )


def build_equivalence() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                'run_id': 'kmeans_001',
                'partition_id': 'partition_001',
                'canonical_labels': '0|0|1|1',
            },
            {
                'run_id': 'spectral_001',
                'partition_id': 'partition_001',
                'canonical_labels': '0|0|1|1',
            },
            {
                'run_id': 'gmm_001',
                'partition_id': 'partition_002',
                'canonical_labels': '0|1|0|1',
            },
        ]
    )


def test_merge_ranking_and_stability_adds_stability_columns():
    merged = merge_ranking_and_stability(
        build_ranking(),
        build_stability(),
    )

    assert 'stability_score' in merged.columns
    assert len(merged) == 3


def test_merge_ranking_and_stability_raises_when_missing_run():
    stability = build_stability()
    stability = stability[stability['run_id'] != 'gmm_001']

    with pytest.raises(ValueError, match='Missing stability results'):
        merge_ranking_and_stability(build_ranking(), stability)


def test_calculate_final_scores_prefers_more_stable_run():
    merged = merge_ranking_and_stability(
        build_ranking(),
        build_stability(),
    )

    final = calculate_final_scores(merged, build_config())

    kmeans_score = final.loc[
        final['run_id'] == 'kmeans_001',
        'final_score',
    ].iloc[0]
    spectral_score = final.loc[
        final['run_id'] == 'spectral_001',
        'final_score',
    ].iloc[0]

    assert spectral_score > kmeans_score


def test_add_partition_information_counts_support():
    merged = merge_ranking_and_stability(
        build_ranking(),
        build_stability(),
    )
    final = calculate_final_scores(merged, build_config())

    enriched = add_partition_information(final, build_equivalence())

    partition = enriched[enriched['partition_id'] == 'partition_001']

    assert partition['n_equivalent_runs'].iloc[0] == 2
    assert partition['n_supporting_algorithms'].iloc[0] == 2
    assert 'kmeans' in partition['supporting_algorithms'].iloc[0]


def test_rank_final_runs_orders_by_final_score():
    final_runs = rank_final_runs(
        ranking=build_ranking(),
        stability=build_stability(),
        equivalence=build_equivalence(),
        experiments_config=build_config(),
    )

    assert final_runs.iloc[0]['final_rank'] == 1
    assert final_runs.iloc[0]['run_id'] == 'spectral_001'
    assert final_runs['final_score'].is_monotonic_decreasing


def test_build_final_partition_ranking_keeps_unique_partitions():
    final_runs = rank_final_runs(
        ranking=build_ranking(),
        stability=build_stability(),
        equivalence=build_equivalence(),
        experiments_config=build_config(),
    )

    partitions = build_final_partition_ranking(final_runs)

    assert len(partitions) == 2
    assert partitions.iloc[0]['partition_id'] == 'partition_001'
    assert partitions.iloc[0]['representative_run_id'] == 'spectral_001'


def test_save_final_outputs_writes_csv_files(tmp_path):
    final_runs = rank_final_runs(
        ranking=build_ranking(),
        stability=build_stability(),
        equivalence=build_equivalence(),
        experiments_config=build_config(tmp_path),
    )
    partitions = build_final_partition_ranking(final_runs)

    run_path, partition_path = save_final_outputs(
        final_runs=final_runs,
        final_partitions=partitions,
        experiments_config=build_config(tmp_path),
    )

    assert run_path.exists()
    assert partition_path.exists()
