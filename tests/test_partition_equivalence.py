import pandas as pd

from src.partition_equivalence import (
    build_partition_signature,
    build_run_partition_map,
    canonicalize_labels,
    save_equivalence_outputs,
    summarize_unique_partitions,
)


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
                'cluster_min_size': 2,
                'cluster_max_size': 2,
                'silhouette': 0.40,
                'davies_bouldin': 0.70,
                'calinski_harabasz': 10.0,
            },
            {
                'rank': 2,
                'run_id': 'agglo_001',
                'algorithm_name': 'agglomerative',
                'ranking_score': 0.88,
                'n_clusters': 2,
                'n_noise': 0,
                'cluster_min_size': 2,
                'cluster_max_size': 2,
                'silhouette': 0.40,
                'davies_bouldin': 0.70,
                'calinski_harabasz': 10.0,
            },
            {
                'rank': 3,
                'run_id': 'gmm_001',
                'algorithm_name': 'gaussian_mixture',
                'ranking_score': 0.70,
                'n_clusters': 2,
                'n_noise': 0,
                'cluster_min_size': 2,
                'cluster_max_size': 2,
                'silhouette': 0.30,
                'davies_bouldin': 0.90,
                'calinski_harabasz': 8.0,
            },
        ]
    )


def build_labels() -> pd.DataFrame:
    rows = []

    run_labels = {
        'kmeans_001': [0, 0, 1, 1],
        'agglo_001': [5, 5, 2, 2],
        'gmm_001': [0, 1, 0, 1],
    }

    for run_id, labels in run_labels.items():
        for index, label in enumerate(labels, start=1):
            rows.append(
                {
                    'run_id': run_id,
                    'AH': f'AH-201{index}',
                    'AH_YEAR': 2010 + index,
                    'cluster_label': label,
                }
            )

    return pd.DataFrame(rows)


def test_canonicalize_labels_remaps_by_first_appearance():
    labels = [7, 7, 3, 3, -1]

    canonical = canonicalize_labels(labels)

    assert canonical == (0, 0, 1, 1, -1)


def test_build_partition_signature_is_invariant_to_label_names():
    signature_a = build_partition_signature([0, 0, 1, 1])
    signature_b = build_partition_signature([5, 5, 2, 2])

    assert signature_a == signature_b


def test_build_partition_signature_changes_when_membership_changes():
    signature_a = build_partition_signature([0, 0, 1, 1])
    signature_b = build_partition_signature([0, 1, 0, 1])

    assert signature_a != signature_b


def test_build_run_partition_map_assigns_same_partition_to_equivalent_runs():
    ranking = build_ranking()
    labels = build_labels()

    run_map = build_run_partition_map(ranking, labels)

    kmeans_partition = run_map.loc[
        run_map['run_id'] == 'kmeans_001',
        'partition_id',
    ].iloc[0]
    agglo_partition = run_map.loc[
        run_map['run_id'] == 'agglo_001',
        'partition_id',
    ].iloc[0]

    assert kmeans_partition == agglo_partition


def test_summarize_unique_partitions_groups_equivalent_runs():
    ranking = build_ranking()
    labels = build_labels()

    ranking_with_partition, summary = summarize_unique_partitions(
        ranking=ranking,
        labels=labels,
    )

    assert len(ranking_with_partition) == 3
    assert len(summary) == 2

    first_partition = summary.iloc[0]

    assert first_partition['n_runs'] == 2
    assert first_partition['best_run_id'] == 'kmeans_001'
    assert 'agglomerative' in first_partition['algorithms']
    assert 'kmeans' in first_partition['algorithms']


def test_save_equivalence_outputs_writes_csv_files(tmp_path):
    ranking = build_ranking()
    labels = build_labels()
    ranking_with_partition, summary = summarize_unique_partitions(
        ranking=ranking,
        labels=labels,
    )
    config = {
        'salidas': {
            'tablas': {
                'benchmark_runs': str(tmp_path / 'benchmark_runs.csv')
            }
        }
    }

    runs_path, summary_path = save_equivalence_outputs(
        ranking_with_partition=ranking_with_partition,
        summary=summary,
        experiments_config=config,
    )

    assert runs_path.exists()
    assert summary_path.exists()
