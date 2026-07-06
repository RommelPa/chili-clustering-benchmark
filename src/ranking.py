from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config_loader import load_experiments_config


REQUIRED_BENCHMARK_COLUMNS = [
    'run_id',
    'algorithm_name',
    'family',
    'role',
    'feature_set_name',
    'scaler_name',
    'status',
    'is_valid_partition',
    'n_clusters',
    'n_noise',
    'noise_ratio',
    'cluster_min_size',
    'cluster_max_size',
    'silhouette',
    'davies_bouldin',
    'calinski_harabasz',
]


def validate_benchmark_columns(benchmark: pd.DataFrame) -> None:
    """Validate required columns for ranking."""
    missing = [
        column for column in REQUIRED_BENCHMARK_COLUMNS
        if column not in benchmark.columns
    ]

    if missing:
        missing_text = ', '.join(missing)
        raise ValueError(f'Missing benchmark columns: {missing_text}')


def to_boolean_series(series: pd.Series) -> pd.Series:
    """Convert boolean-like values to a boolean Series."""
    if pd.api.types.is_bool_dtype(series):
        return series

    return series.astype(str).str.lower().isin(['true', '1', 'yes'])


def filter_rankable_runs(benchmark: pd.DataFrame) -> pd.DataFrame:
    """Keep only successful, valid runs with complete internal metrics."""
    validate_benchmark_columns(benchmark)

    valid_partition = to_boolean_series(benchmark['is_valid_partition'])

    rankable = benchmark[
        (benchmark['status'] == 'success')
        & valid_partition
        & benchmark['silhouette'].notna()
        & benchmark['davies_bouldin'].notna()
        & benchmark['calinski_harabasz'].notna()
    ].copy()

    if rankable.empty:
        raise ValueError('No rankable benchmark runs were found.')

    return rankable


def min_max_scale(
    values: pd.Series,
    higher_is_better: bool = True,
) -> pd.Series:
    """Scale a metric to [0, 1]."""
    numeric_values = values.astype(float)
    min_value = numeric_values.min()
    max_value = numeric_values.max()

    if np.isclose(max_value, min_value):
        return pd.Series(1.0, index=values.index)

    scaled = (numeric_values - min_value) / (max_value - min_value)

    if higher_is_better:
        return scaled

    return 1.0 - scaled


def add_normalized_metrics(rankable: pd.DataFrame) -> pd.DataFrame:
    """Add normalized metric columns for multicriteria ranking."""
    ranked = rankable.copy()

    ranked['silhouette_norm'] = min_max_scale(
        ranked['silhouette'],
        higher_is_better=True,
    )
    ranked['davies_bouldin_norm'] = min_max_scale(
        ranked['davies_bouldin'],
        higher_is_better=False,
    )
    ranked['calinski_harabasz_norm'] = min_max_scale(
        ranked['calinski_harabasz'],
        higher_is_better=True,
    )

    return ranked


def add_penalty_columns(rankable: pd.DataFrame) -> pd.DataFrame:
    """Add penalty columns for noise and cluster imbalance."""
    ranked = rankable.copy()

    ranked['noise_penalty_value'] = ranked['noise_ratio'].astype(float).clip(
        lower=0.0,
    )

    ranked['cluster_imbalance'] = np.where(
        ranked['cluster_max_size'].astype(float) > 0,
        1.0
        - (
            ranked['cluster_min_size'].astype(float)
            / ranked['cluster_max_size'].astype(float)
        ),
        1.0,
    )

    ranked['cluster_size_penalty_value'] = ranked['cluster_imbalance'].clip(
        lower=0.0,
        upper=1.0,
    )

    return ranked


def calculate_ranking_score(
    ranked: pd.DataFrame,
    experiments_config: dict[str, Any],
) -> pd.DataFrame:
    """Calculate preliminary ranking score without stability."""
    result = ranked.copy()
    weights = experiments_config['ranking']['pesos']

    positive_weights = {
        'silhouette_norm': float(weights['silhouette']),
        'davies_bouldin_norm': float(weights['davies_bouldin']),
        'calinski_harabasz_norm': float(weights['calinski_harabasz']),
    }
    total_positive_weight = sum(positive_weights.values())

    result['internal_score'] = 0.0

    for metric_name, metric_weight in positive_weights.items():
        normalized_weight = metric_weight / total_positive_weight
        result['internal_score'] += result[metric_name] * normalized_weight

    noise_weight = float(weights.get('penalizacion_ruido', 0.0))
    cluster_weight = float(weights.get('penalizacion_cluster_pequeno', 0.0))

    result['ranking_score'] = (
        result['internal_score']
        - noise_weight * result['noise_penalty_value']
        - cluster_weight * result['cluster_size_penalty_value']
    ).clip(lower=0.0, upper=1.0)

    result['ranking_stage'] = 'internal_pre_stability'

    return result


def rank_benchmark_runs(
    benchmark: pd.DataFrame,
    experiments_config: dict[str, Any],
) -> pd.DataFrame:
    """Rank successful and valid benchmark runs."""
    rankable = filter_rankable_runs(benchmark)
    ranked = add_normalized_metrics(rankable)
    ranked = add_penalty_columns(ranked)
    ranked = calculate_ranking_score(ranked, experiments_config)

    ranked = ranked.sort_values(
        by=[
            'ranking_score',
            'silhouette',
            'davies_bouldin',
            'calinski_harabasz',
        ],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)

    ranked.insert(0, 'rank', range(1, len(ranked) + 1))

    return ranked


def summarize_top_by_algorithm(ranked: pd.DataFrame) -> pd.DataFrame:
    """Keep the best-ranked run per algorithm."""
    return (
        ranked.sort_values('rank')
        .groupby('algorithm_name', as_index=False)
        .head(1)
        .sort_values('rank')
        .reset_index(drop=True)
    )


def get_ranking_output_paths(
    experiments_config: dict[str, Any],
) -> tuple[Path, Path]:
    """Return ranking output paths."""
    benchmark_path = Path(
        experiments_config['salidas']['tablas']['benchmark_runs']
    )
    output_dir = benchmark_path.parent

    return (
        output_dir / 'ranking_internal.csv',
        output_dir / 'ranking_top_by_algorithm.csv',
    )


def save_ranking_outputs(
    ranked: pd.DataFrame,
    top_by_algorithm: pd.DataFrame,
    experiments_config: dict[str, Any],
) -> tuple[Path, Path]:
    """Save ranking outputs."""
    ranking_path, top_path = get_ranking_output_paths(experiments_config)
    ranking_path.parent.mkdir(parents=True, exist_ok=True)

    ranked.to_csv(ranking_path, index=False, encoding='utf-8')
    top_by_algorithm.to_csv(top_path, index=False, encoding='utf-8')

    return ranking_path, top_path


def run_ranking(experiments_config: dict[str, Any]) -> pd.DataFrame:
    """Run preliminary ranking from benchmark output table."""
    benchmark_path = Path(
        experiments_config['salidas']['tablas']['benchmark_runs']
    )

    benchmark = pd.read_csv(benchmark_path)
    ranked = rank_benchmark_runs(benchmark, experiments_config)
    top_by_algorithm = summarize_top_by_algorithm(ranked)

    ranking_path, top_path = save_ranking_outputs(
        ranked=ranked,
        top_by_algorithm=top_by_algorithm,
        experiments_config=experiments_config,
    )

    print('[ranking] Ranking preliminar generado correctamente')
    print(f'[ranking] Corridas rankeadas: {len(ranked)}')
    print(f'[ranking] Tabla ranking: {ranking_path}')
    print(f'[ranking] Tabla mejores por algoritmo: {top_path}')
    print('[ranking] Top 10 preliminar:')

    columns = [
        'rank',
        'run_id',
        'algorithm_name',
        'n_clusters',
        'ranking_score',
        'silhouette',
        'davies_bouldin',
        'calinski_harabasz',
        'n_noise',
        'cluster_min_size',
    ]

    print(ranked[columns].head(10).to_string(index=False))

    return ranked


if __name__ == '__main__':
    experiments_config = load_experiments_config()
    run_ranking(experiments_config)
