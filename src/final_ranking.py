from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config_loader import load_experiments_config


REQUIRED_RANKING_COLUMNS = [
    'rank',
    'run_id',
    'algorithm_name',
    'ranking_score',
    'n_clusters',
    'n_noise',
    'cluster_min_size',
    'cluster_max_size',
    'silhouette',
    'davies_bouldin',
    'calinski_harabasz',
    'silhouette_norm',
    'davies_bouldin_norm',
    'calinski_harabasz_norm',
    'noise_penalty_value',
    'cluster_size_penalty_value',
]

REQUIRED_STABILITY_COLUMNS = [
    'run_id',
    'stability_score',
    'stability_intra',
    'stability_inter',
    'stability_gap',
    'n_successful_iterations',
]

REQUIRED_EQUIVALENCE_COLUMNS = [
    'run_id',
    'partition_id',
    'canonical_labels',
]


def validate_required_columns(
    data: pd.DataFrame,
    required_columns: list[str],
    table_name: str,
) -> None:
    """Validate that a table has all required columns."""
    missing = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing:
        missing_text = ', '.join(missing)
        raise ValueError(f'Missing columns in {table_name}: {missing_text}')


def merge_ranking_and_stability(
    ranking: pd.DataFrame,
    stability: pd.DataFrame,
) -> pd.DataFrame:
    """Merge preliminary ranking with stability results."""
    validate_required_columns(
        ranking,
        REQUIRED_RANKING_COLUMNS,
        'ranking',
    )
    validate_required_columns(
        stability,
        REQUIRED_STABILITY_COLUMNS,
        'stability',
    )

    stability_subset = stability[REQUIRED_STABILITY_COLUMNS].copy()

    merged = ranking.merge(
        stability_subset,
        on='run_id',
        how='left',
    )

    if merged['stability_score'].isna().any():
        missing_runs = merged.loc[
            merged['stability_score'].isna(),
            'run_id',
        ].tolist()
        raise ValueError(f'Missing stability results for runs: {missing_runs}')

    return merged


def calculate_final_scores(
    ranked_runs: pd.DataFrame,
    experiments_config: dict[str, Any],
) -> pd.DataFrame:
    """Calculate final score using internal metrics and stability."""
    result = ranked_runs.copy()
    weights = experiments_config['ranking']['pesos']

    positive_weights = {
        'silhouette_norm': float(weights['silhouette']),
        'davies_bouldin_norm': float(weights['davies_bouldin']),
        'calinski_harabasz_norm': float(weights['calinski_harabasz']),
        'stability_score': float(weights['estabilidad']),
    }
    positive_total = sum(positive_weights.values())

    result['positive_score'] = 0.0

    for column, weight in positive_weights.items():
        values = result[column].astype(float).clip(lower=0.0, upper=1.0)
        result['positive_score'] += values * weight

    result['positive_score'] = result['positive_score'] / positive_total

    noise_weight = float(weights.get('penalizacion_ruido', 0.0))
    cluster_weight = float(weights.get('penalizacion_cluster_pequeno', 0.0))

    result['final_penalty'] = (
        noise_weight * result['noise_penalty_value'].astype(float)
        + cluster_weight * result['cluster_size_penalty_value'].astype(float)
    )

    result['final_score'] = (
        result['positive_score'] - result['final_penalty']
    ).clip(lower=0.0, upper=1.0)

    result['ranking_stage'] = 'internal_plus_stability'

    return result


def add_partition_information(
    ranked_runs: pd.DataFrame,
    equivalence: pd.DataFrame,
) -> pd.DataFrame:
    """Add equivalent partition identifiers and support counts."""
    validate_required_columns(
        equivalence,
        REQUIRED_EQUIVALENCE_COLUMNS,
        'partition_equivalence',
    )

    equivalence_subset = equivalence[REQUIRED_EQUIVALENCE_COLUMNS].copy()

    enriched = ranked_runs.merge(
        equivalence_subset,
        on='run_id',
        how='left',
    )

    if enriched['partition_id'].isna().any():
        missing_runs = enriched.loc[
            enriched['partition_id'].isna(),
            'run_id',
        ].tolist()
        raise ValueError(f'Missing partition IDs for runs: {missing_runs}')

    support = (
        enriched.groupby('partition_id')
        .agg(
            n_equivalent_runs=('run_id', 'count'),
            n_supporting_algorithms=('algorithm_name', 'nunique'),
        )
        .reset_index()
    )

    algorithms = (
        enriched.groupby('partition_id')['algorithm_name']
        .apply(lambda values: ', '.join(sorted(values.astype(str).unique())))
        .reset_index(name='supporting_algorithms')
    )

    support = support.merge(algorithms, on='partition_id', how='left')

    return enriched.merge(
        support,
        on='partition_id',
        how='left',
    )


def rank_final_runs(
    ranking: pd.DataFrame,
    stability: pd.DataFrame,
    equivalence: pd.DataFrame,
    experiments_config: dict[str, Any],
) -> pd.DataFrame:
    """Build final run-level ranking."""
    ranked_runs = merge_ranking_and_stability(ranking, stability)
    ranked_runs = calculate_final_scores(ranked_runs, experiments_config)
    ranked_runs = add_partition_information(ranked_runs, equivalence)

    ranked_runs = ranked_runs.sort_values(
        by=[
            'final_score',
            'stability_score',
            'ranking_score',
            'n_supporting_algorithms',
            'n_equivalent_runs',
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    ranked_runs.insert(0, 'final_rank', range(1, len(ranked_runs) + 1))

    return ranked_runs


def build_final_partition_ranking(final_runs: pd.DataFrame) -> pd.DataFrame:
    """Build final ranking at unique partition level."""
    summary_rows: list[dict[str, Any]] = []

    for partition_id, group in final_runs.groupby('partition_id'):
        ordered = group.sort_values(
            by=[
                'final_score',
                'stability_score',
                'ranking_score',
                'rank',
            ],
            ascending=[False, False, False, True],
        )
        best = ordered.iloc[0]

        summary_rows.append(
            {
                'partition_id': partition_id,
                'representative_run_id': best['run_id'],
                'representative_algorithm': best['algorithm_name'],
                'final_score': float(best['final_score']),
                'positive_score': float(best['positive_score']),
                'final_penalty': float(best['final_penalty']),
                'ranking_score': float(best['ranking_score']),
                'stability_score': float(best['stability_score']),
                'stability_intra': float(best['stability_intra']),
                'stability_inter': float(best['stability_inter']),
                'stability_gap': float(best['stability_gap']),
                'best_internal_rank': int(group['rank'].min()),
                'n_clusters': int(best['n_clusters']),
                'n_noise': int(best['n_noise']),
                'cluster_min_size': int(best['cluster_min_size']),
                'cluster_max_size': int(best['cluster_max_size']),
                'silhouette': float(best['silhouette']),
                'davies_bouldin': float(best['davies_bouldin']),
                'calinski_harabasz': float(best['calinski_harabasz']),
                'n_equivalent_runs': int(best['n_equivalent_runs']),
                'n_supporting_algorithms': int(best['n_supporting_algorithms']),
                'supporting_algorithms': best['supporting_algorithms'],
                'canonical_labels': best['canonical_labels'],
            }
        )

    partitions = pd.DataFrame(summary_rows).sort_values(
        by=[
            'final_score',
            'stability_score',
            'ranking_score',
            'n_supporting_algorithms',
            'n_equivalent_runs',
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    partitions.insert(
        0,
        'final_partition_rank',
        range(1, len(partitions) + 1),
    )

    return partitions


def get_final_output_paths(
    experiments_config: dict[str, Any],
) -> tuple[Path, Path]:
    """Return final ranking output paths."""
    benchmark_path = Path(
        experiments_config['salidas']['tablas']['benchmark_runs']
    )
    output_dir = benchmark_path.parent

    return (
        output_dir / 'final_run_ranking.csv',
        output_dir / 'final_partition_ranking.csv',
    )


def save_final_outputs(
    final_runs: pd.DataFrame,
    final_partitions: pd.DataFrame,
    experiments_config: dict[str, Any],
) -> tuple[Path, Path]:
    """Save final ranking outputs."""
    run_path, partition_path = get_final_output_paths(experiments_config)
    run_path.parent.mkdir(parents=True, exist_ok=True)

    final_runs.to_csv(run_path, index=False, encoding='utf-8')
    final_partitions.to_csv(partition_path, index=False, encoding='utf-8')

    return run_path, partition_path


def load_required_outputs(
    experiments_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load ranking, stability and partition equivalence outputs."""
    benchmark_path = Path(
        experiments_config['salidas']['tablas']['benchmark_runs']
    )
    output_dir = benchmark_path.parent

    ranking = pd.read_csv(output_dir / 'ranking_internal.csv')
    stability = pd.read_csv(output_dir / 'stability_runs.csv')
    equivalence = pd.read_csv(output_dir / 'partition_equivalence_runs.csv')

    return ranking, stability, equivalence


def run_final_ranking(experiments_config: dict[str, Any]) -> pd.DataFrame:
    """Run final ranking integration."""
    ranking, stability, equivalence = load_required_outputs(experiments_config)

    final_runs = rank_final_runs(
        ranking=ranking,
        stability=stability,
        equivalence=equivalence,
        experiments_config=experiments_config,
    )
    final_partitions = build_final_partition_ranking(final_runs)

    run_path, partition_path = save_final_outputs(
        final_runs=final_runs,
        final_partitions=final_partitions,
        experiments_config=experiments_config,
    )

    print('[final_ranking] Ranking final integrado generado')
    print(f'[final_ranking] Corridas finales: {len(final_runs)}')
    print(f'[final_ranking] Particiones únicas: {len(final_partitions)}')
    print(f'[final_ranking] Tabla corridas: {run_path}')
    print(f'[final_ranking] Tabla particiones: {partition_path}')
    print('[final_ranking] Top particiones finales:')

    columns = [
        'final_partition_rank',
        'partition_id',
        'representative_run_id',
        'representative_algorithm',
        'final_score',
        'ranking_score',
        'stability_score',
        'n_clusters',
        'n_equivalent_runs',
        'n_supporting_algorithms',
        'supporting_algorithms',
    ]

    print(final_partitions[columns].head(10).to_string(index=False))

    return final_partitions


if __name__ == '__main__':
    experiments_config = load_experiments_config()
    run_final_ranking(experiments_config)

