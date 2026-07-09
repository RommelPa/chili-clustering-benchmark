from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_BENCHMARK_COLUMNS = {
    'run_id',
    'algorithm_name',
    'family',
    'role',
    'status',
    'is_valid_partition',
}

REQUIRED_FINAL_RUN_COLUMNS = {
    'final_rank',
    'run_id',
    'algorithm_name',
    'family',
    'role',
    'partition_id',
    'final_score',
    'ranking_score',
    'stability_score',
    'n_clusters',
    'silhouette',
    'davies_bouldin',
    'calinski_harabasz',
}


def get_table_output_dir(experiments_config: dict[str, Any]) -> Path:
    """Return output table directory from experiment configuration."""
    benchmark_path = Path(
        experiments_config['salidas']['tablas']['benchmark_runs']
    )

    return benchmark_path.parent


def validate_required_columns(
    data: pd.DataFrame,
    required_columns: set[str],
    table_name: str,
) -> None:
    """Validate that a table contains all required columns."""
    missing = sorted(required_columns - set(data.columns))

    if missing:
        raise ValueError(f'Missing columns in {table_name}: {missing}')


def to_bool_series(values: pd.Series) -> pd.Series:
    """Convert mixed boolean-like values to booleans."""
    return values.astype(str).str.lower().isin(['true', '1', 'yes', 'y'])


def summarize_algorithm_configurations(
    benchmark: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize evaluated configurations by algorithm."""
    validate_required_columns(
        benchmark,
        REQUIRED_BENCHMARK_COLUMNS,
        'benchmark_runs',
    )

    data = benchmark.copy()
    data['is_successful_run'] = data['status'].astype(str).eq('success')
    data['is_valid_partition_bool'] = to_bool_series(
        data['is_valid_partition']
    )

    summary = (
        data.groupby(['algorithm_name', 'family', 'role'], dropna=False)
        .agg(
            n_configurations=('run_id', 'count'),
            n_successful_runs=('is_successful_run', 'sum'),
            n_valid_partitions=('is_valid_partition_bool', 'sum'),
        )
        .reset_index()
    )

    summary['valid_partition_rate'] = (
        summary['n_valid_partitions'] / summary['n_configurations']
    )

    return summary.sort_values('algorithm_name').reset_index(drop=True)


def select_best_run_by_algorithm(final_runs: pd.DataFrame) -> pd.DataFrame:
    """Select the best final-ranked run for each algorithm."""
    validate_required_columns(
        final_runs,
        REQUIRED_FINAL_RUN_COLUMNS,
        'final_run_ranking',
    )

    sort_columns = [
        'algorithm_name',
        'final_score',
        'stability_score',
        'ranking_score',
        'final_rank',
    ]
    ascending = [True, False, False, False, True]

    ordered = final_runs.sort_values(
        by=sort_columns,
        ascending=ascending,
    )

    best_runs = (
        ordered.groupby('algorithm_name', as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    return best_runs.sort_values(
        by='final_score',
        ascending=False,
    ).reset_index(drop=True)


def build_algorithm_performance_summary(
    benchmark: pd.DataFrame,
    final_runs: pd.DataFrame,
) -> pd.DataFrame:
    """Build final algorithm-level performance summary."""
    configuration_summary = summarize_algorithm_configurations(benchmark)
    best_runs = select_best_run_by_algorithm(final_runs)

    final_counts = (
        final_runs.groupby('algorithm_name')
        .agg(n_final_ranked_runs=('run_id', 'count'))
        .reset_index()
    )

    best_columns = [
        'algorithm_name',
        'run_id',
        'partition_id',
        'final_rank',
        'final_score',
        'ranking_score',
        'stability_score',
        'n_clusters',
        'silhouette',
        'davies_bouldin',
        'calinski_harabasz',
    ]

    best_runs_compact = best_runs[best_columns].rename(
        columns={
            'run_id': 'best_run_id',
            'partition_id': 'best_partition_id',
            'final_rank': 'best_final_rank',
            'final_score': 'best_final_score',
            'ranking_score': 'best_ranking_score',
            'stability_score': 'best_stability_score',
            'n_clusters': 'best_n_clusters',
            'silhouette': 'best_silhouette',
            'davies_bouldin': 'best_davies_bouldin',
            'calinski_harabasz': 'best_calinski_harabasz',
        }
    )

    summary = configuration_summary.merge(
        final_counts,
        on='algorithm_name',
        how='left',
    ).merge(
        best_runs_compact,
        on='algorithm_name',
        how='left',
    )

    summary['n_final_ranked_runs'] = (
        summary['n_final_ranked_runs'].fillna(0).astype(int)
    )

    return summary.sort_values(
        by=['best_final_score', 'n_valid_partitions'],
        ascending=[False, False],
        na_position='last',
    ).reset_index(drop=True)


def get_algorithm_analysis_output_paths(
    experiments_config: dict[str, Any],
) -> tuple[Path, Path]:
    """Return output paths for algorithm analysis tables."""
    output_dir = get_table_output_dir(experiments_config)

    return (
        output_dir / 'algorithm_performance_summary.csv',
        output_dir / 'algorithm_best_runs.csv',
    )


def save_algorithm_analysis_outputs(
    algorithm_summary: pd.DataFrame,
    best_runs: pd.DataFrame,
    experiments_config: dict[str, Any],
) -> tuple[Path, Path]:
    """Save algorithm analysis tables."""
    summary_path, best_runs_path = get_algorithm_analysis_output_paths(
        experiments_config
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)

    algorithm_summary.to_csv(summary_path, index=False, encoding='utf-8')
    best_runs.to_csv(best_runs_path, index=False, encoding='utf-8')

    return summary_path, best_runs_path


def load_required_outputs(
    experiments_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load benchmark and final run ranking outputs."""
    output_dir = get_table_output_dir(experiments_config)

    benchmark = pd.read_csv(output_dir / 'benchmark_runs.csv')
    final_runs = pd.read_csv(output_dir / 'final_run_ranking.csv')

    return benchmark, final_runs


def run_algorithm_analysis(
    experiments_config: dict[str, Any],
) -> pd.DataFrame:
    """Run algorithm-level analysis."""
    benchmark, final_runs = load_required_outputs(experiments_config)

    algorithm_summary = build_algorithm_performance_summary(
        benchmark=benchmark,
        final_runs=final_runs,
    )
    best_runs = select_best_run_by_algorithm(final_runs)

    summary_path, best_runs_path = save_algorithm_analysis_outputs(
        algorithm_summary=algorithm_summary,
        best_runs=best_runs,
        experiments_config=experiments_config,
    )

    print('[algorithm_analysis] Análisis por algoritmo generado')
    print(f'[algorithm_analysis] Tabla resumen: {summary_path}')
    print(f'[algorithm_analysis] Mejores corridas: {best_runs_path}')
    print('[algorithm_analysis] Resumen:')
    print(
        algorithm_summary[
            [
                'algorithm_name',
                'n_configurations',
                'n_valid_partitions',
                'best_run_id',
                'best_partition_id',
                'best_final_score',
                'best_stability_score',
                'best_n_clusters',
            ]
        ].round(4).to_string(index=False)
    )

    return algorithm_summary


if __name__ == '__main__':
    from src.config_loader import load_experiments_config

    experiments_config = load_experiments_config()
    run_algorithm_analysis(experiments_config)
