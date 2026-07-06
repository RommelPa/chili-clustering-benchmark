from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config_loader import load_experiments_config


def canonicalize_labels(labels: list[int]) -> tuple[int, ...]:
    """Convert arbitrary cluster labels into a canonical sequence.

    Noise label -1 is preserved.
    Non-noise labels are remapped by order of first appearance.
    """
    label_map: dict[int, int] = {}
    next_label = 0
    canonical: list[int] = []

    for label in labels:
        label_int = int(label)

        if label_int == -1:
            canonical.append(-1)
            continue

        if label_int not in label_map:
            label_map[label_int] = next_label
            next_label += 1

        canonical.append(label_map[label_int])

    return tuple(canonical)


def build_partition_signature(labels: list[int]) -> str:
    """Build a stable text signature for a partition."""
    canonical = canonicalize_labels(labels)

    return '|'.join(str(label) for label in canonical)


def get_ranked_run_labels(
    ranking: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Keep labels only for ranked runs."""
    ranked_run_ids = set(ranking['run_id'].astype(str))

    ranked_labels = labels[
        labels['run_id'].astype(str).isin(ranked_run_ids)
    ].copy()

    if ranked_labels.empty:
        raise ValueError('No labels found for ranked runs.')

    return ranked_labels


def build_run_partition_map(
    ranking: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Assign each ranked run to a canonical partition signature."""
    ranked_labels = get_ranked_run_labels(ranking, labels)

    rows: list[dict[str, Any]] = []

    for run_id, group in ranked_labels.groupby('run_id'):
        ordered = group.sort_values('AH_YEAR')
        labels_list = ordered['cluster_label'].astype(int).tolist()
        canonical = canonicalize_labels(labels_list)

        rows.append(
            {
                'run_id': run_id,
                'partition_signature': build_partition_signature(labels_list),
                'canonical_labels': '|'.join(str(label) for label in canonical),
            }
        )

    run_map = pd.DataFrame(rows)

    unique_signatures = run_map['partition_signature'].drop_duplicates()
    signature_to_id = {
        signature: f'partition_{index:03d}'
        for index, signature in enumerate(unique_signatures, start=1)
    }

    run_map['partition_id'] = run_map['partition_signature'].map(
        signature_to_id,
    )

    return run_map


def summarize_unique_partitions(
    ranking: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize unique partitions among ranked benchmark runs."""
    run_map = build_run_partition_map(ranking, labels)

    ranking_with_partition = ranking.merge(
        run_map,
        on='run_id',
        how='inner',
    )

    summary_rows: list[dict[str, Any]] = []

    for partition_id, group in ranking_with_partition.groupby('partition_id'):
        ordered = group.sort_values('rank')
        best = ordered.iloc[0]

        algorithms = ', '.join(
            sorted(ordered['algorithm_name'].astype(str).unique())
        )

        summary_rows.append(
            {
                'partition_id': partition_id,
                'n_runs': len(ordered),
                'algorithms': algorithms,
                'best_rank': int(best['rank']),
                'best_run_id': best['run_id'],
                'best_algorithm': best['algorithm_name'],
                'ranking_score': float(best['ranking_score']),
                'n_clusters': int(best['n_clusters']),
                'n_noise': int(best['n_noise']),
                'cluster_min_size': int(best['cluster_min_size']),
                'cluster_max_size': int(best['cluster_max_size']),
                'silhouette': float(best['silhouette']),
                'davies_bouldin': float(best['davies_bouldin']),
                'calinski_harabasz': float(best['calinski_harabasz']),
                'canonical_labels': best['canonical_labels'],
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        by=['best_rank'],
    ).reset_index(drop=True)

    return ranking_with_partition, summary


def get_equivalence_output_paths(
    experiments_config: dict[str, Any],
) -> tuple[Path, Path]:
    """Return output paths for partition equivalence tables."""
    benchmark_path = Path(
        experiments_config['salidas']['tablas']['benchmark_runs']
    )
    output_dir = benchmark_path.parent

    return (
        output_dir / 'partition_equivalence_runs.csv',
        output_dir / 'partition_equivalence_summary.csv',
    )


def save_equivalence_outputs(
    ranking_with_partition: pd.DataFrame,
    summary: pd.DataFrame,
    experiments_config: dict[str, Any],
) -> tuple[Path, Path]:
    """Save partition equivalence outputs."""
    runs_path, summary_path = get_equivalence_output_paths(experiments_config)
    runs_path.parent.mkdir(parents=True, exist_ok=True)

    ranking_with_partition.to_csv(runs_path, index=False, encoding='utf-8')
    summary.to_csv(summary_path, index=False, encoding='utf-8')

    return runs_path, summary_path


def run_partition_equivalence(
    experiments_config: dict[str, Any],
) -> pd.DataFrame:
    """Detect equivalent partitions from ranking and labels outputs."""
    benchmark_path = Path(
        experiments_config['salidas']['tablas']['benchmark_runs']
    )
    output_dir = benchmark_path.parent

    ranking_path = output_dir / 'ranking_internal.csv'
    labels_path = output_dir / 'benchmark_labels.csv'

    ranking = pd.read_csv(ranking_path)
    labels = pd.read_csv(labels_path)

    ranking_with_partition, summary = summarize_unique_partitions(
        ranking=ranking,
        labels=labels,
    )

    runs_path, summary_path = save_equivalence_outputs(
        ranking_with_partition=ranking_with_partition,
        summary=summary,
        experiments_config=experiments_config,
    )

    print('[partition_equivalence] Particiones equivalentes detectadas')
    print(f'[partition_equivalence] Corridas rankeadas: {len(ranking)}')
    print(f'[partition_equivalence] Particiones únicas: {len(summary)}')
    print(f'[partition_equivalence] Tabla corridas: {runs_path}')
    print(f'[partition_equivalence] Tabla resumen: {summary_path}')
    print('[partition_equivalence] Top particiones únicas:')

    columns = [
        'partition_id',
        'n_runs',
        'algorithms',
        'best_rank',
        'best_run_id',
        'best_algorithm',
        'ranking_score',
        'n_clusters',
        'cluster_min_size',
        'silhouette',
    ]

    print(summary[columns].head(10).to_string(index=False))

    return summary


if __name__ == '__main__':
    experiments_config = load_experiments_config()
    run_partition_equivalence(experiments_config)
