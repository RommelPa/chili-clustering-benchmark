from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
from sklearn.metrics import adjusted_rand_score, silhouette_samples

from src.config_loader import load_experiments_config, load_pipeline_config
from src.experiment_matrix import get_feature_sets, get_scalers
from src.metrics import safe_internal_metrics
from src.stability import get_project_random_state


DEFAULT_N_BOOTSTRAP = 10000
DEFAULT_ALPHA = 0.05


def get_table_output_dir(experiments_config: dict[str, Any]) -> Path:
    """Return output table directory from experiment configuration."""
    benchmark_path = Path(
        experiments_config['salidas']['tablas']['benchmark_runs']
    )

    return benchmark_path.parent


def get_statistical_settings(
    experiments_config: dict[str, Any],
) -> tuple[int, float]:
    """Return configured bootstrap size and significance level, with defaults."""
    settings = experiments_config.get('pruebas_estadisticas', {})

    n_bootstrap = int(settings.get('n_bootstrap', DEFAULT_N_BOOTSTRAP))
    alpha = float(settings.get('nivel_significancia', DEFAULT_ALPHA))

    return n_bootstrap, alpha


def parse_canonical_labels(canonical_labels: str) -> np.ndarray:
    """Parse a pipe-separated canonical label string into an integer array."""
    return np.array(
        [int(label) for label in str(canonical_labels).split('|')],
        dtype=int,
    )


def load_annual_matrix(
    pipeline_config: dict[str, Any],
    experiments_config: dict[str, Any],
) -> np.ndarray:
    """Load the configured annual feature matrix, ordered by AH_YEAR.

    The current experimental design uses a single feature set and scaler
    combination for every benchmarked configuration, so this loader assumes
    exactly one of each is configured.
    """
    feature_sets = get_feature_sets(experiments_config)
    scalers = get_scalers(experiments_config)

    if len(feature_sets) != 1 or len(scalers) != 1:
        raise ValueError(
            'Statistical tests currently assume a single feature set '
            'and scaler configuration.'
        )

    feature_set_name = next(iter(feature_sets))
    variables = feature_sets[feature_set_name]
    scaler_name = scalers[0]

    table_dir = Path(pipeline_config['rutas']['carpeta_tablas'])
    matrix_path = table_dir / f'matrix_{feature_set_name}_{scaler_name}.csv'
    matrix_frame = pd.read_csv(matrix_path).sort_values('AH_YEAR')

    return matrix_frame[variables].to_numpy(dtype=float)


def load_final_partitions(experiments_config: dict[str, Any]) -> pd.DataFrame:
    """Load the final unique-partition ranking table, ordered by rank."""
    output_dir = get_table_output_dir(experiments_config)
    partitions = pd.read_csv(output_dir / 'final_partition_ranking.csv')

    return partitions.sort_values('final_partition_rank').reset_index(drop=True)


def load_stability_iterations(experiments_config: dict[str, Any]) -> pd.DataFrame:
    """Load per-iteration subsampling stability results for benchmark runs."""
    output_dir = get_table_output_dir(experiments_config)

    return pd.read_csv(output_dir / 'stability_iterations.csv')


def apply_benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Adjust p-values with the Benjamini-Hochberg false discovery rate."""
    n_tests = len(p_values)

    if n_tests == 0:
        return np.array([])

    order = np.argsort(p_values)
    ranked_p_values = p_values[order]

    adjusted = ranked_p_values * n_tests / (np.arange(n_tests) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    result = np.empty(n_tests)
    result[order] = adjusted

    return result


def bootstrap_metric_differences(
    values: np.ndarray,
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    n_bootstrap: int,
    random_state: int,
) -> dict[str, np.ndarray]:
    """Build paired bootstrap distributions of internal metric differences.

    Hydrological years are resampled with replacement, and the same resampled
    indices are used to relabel both partitions in each replicate, so the
    comparison stays paired at the observation level. Replicates where a
    partition collapses to fewer than two effective clusters are discarded.
    Differences are oriented so that a positive value favors partition A
    (higher silhouette, lower Davies-Bouldin, higher Calinski-Harabasz).
    """
    rng = np.random.default_rng(random_state)
    n_items = len(labels_a)

    silhouette_diffs: list[float] = []
    davies_bouldin_diffs: list[float] = []
    calinski_harabasz_diffs: list[float] = []

    for _ in range(n_bootstrap):
        sample_indices = rng.integers(0, n_items, size=n_items)

        metrics_a = safe_internal_metrics(
            values[sample_indices],
            labels_a[sample_indices],
        )
        metrics_b = safe_internal_metrics(
            values[sample_indices],
            labels_b[sample_indices],
        )

        if not (
            np.isfinite(metrics_a['silhouette'])
            and np.isfinite(metrics_b['silhouette'])
        ):
            continue

        silhouette_diffs.append(
            metrics_a['silhouette'] - metrics_b['silhouette']
        )
        davies_bouldin_diffs.append(
            metrics_b['davies_bouldin'] - metrics_a['davies_bouldin']
        )
        calinski_harabasz_diffs.append(
            metrics_a['calinski_harabasz'] - metrics_b['calinski_harabasz']
        )

    return {
        'silhouette_diff': np.array(silhouette_diffs),
        'davies_bouldin_diff': np.array(davies_bouldin_diffs),
        'calinski_harabasz_diff': np.array(calinski_harabasz_diffs),
    }


def bootstrap_p_value(diffs: np.ndarray) -> float:
    """Two-sided bootstrap p-value against a zero-difference null hypothesis."""
    if diffs.size == 0:
        return float('nan')

    share_at_or_below_zero = float(np.mean(diffs <= 0.0))
    share_at_or_above_zero = float(np.mean(diffs >= 0.0))

    p_value = 2.0 * min(share_at_or_below_zero, share_at_or_above_zero)

    return float(min(p_value, 1.0))


def bootstrap_confidence_interval(
    diffs: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a difference distribution."""
    if diffs.size == 0:
        return (float('nan'), float('nan'))

    alpha = 1.0 - confidence
    lower = float(np.percentile(diffs, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(diffs, 100.0 * (1.0 - alpha / 2.0)))

    return lower, upper


def build_bootstrap_comparison_table(
    values: np.ndarray,
    partitions: pd.DataFrame,
    random_state: int,
    n_bootstrap: int,
    alpha: float,
) -> pd.DataFrame:
    """Compare the top-ranked partition against every other retained partition."""
    reference_row = partitions.iloc[0]
    reference_labels = parse_canonical_labels(reference_row['canonical_labels'])

    rows: list[dict[str, Any]] = []

    for offset, (_, row) in enumerate(partitions.iloc[1:].iterrows(), start=1):
        candidate_labels = parse_canonical_labels(row['canonical_labels'])

        diffs = bootstrap_metric_differences(
            values=values,
            labels_a=reference_labels,
            labels_b=candidate_labels,
            n_bootstrap=n_bootstrap,
            random_state=random_state + offset,
        )

        silhouette_p_value = bootstrap_p_value(diffs['silhouette_diff'])
        silhouette_ci_low, silhouette_ci_high = bootstrap_confidence_interval(
            diffs['silhouette_diff']
        )

        rows.append(
            {
                'reference_partition_id': reference_row['partition_id'],
                'reference_run_id': reference_row['representative_run_id'],
                'candidate_partition_id': row['partition_id'],
                'candidate_run_id': row['representative_run_id'],
                'candidate_final_rank': int(row['final_partition_rank']),
                'n_valid_bootstrap_samples': int(diffs['silhouette_diff'].size),
                'observed_silhouette_diff': float(
                    reference_row['silhouette'] - row['silhouette']
                ),
                'silhouette_diff_ci_low': silhouette_ci_low,
                'silhouette_diff_ci_high': silhouette_ci_high,
                'silhouette_bootstrap_p_value': silhouette_p_value,
                'observed_davies_bouldin_diff': float(
                    row['davies_bouldin'] - reference_row['davies_bouldin']
                ),
                'davies_bouldin_bootstrap_p_value': bootstrap_p_value(
                    diffs['davies_bouldin_diff']
                ),
                'observed_calinski_harabasz_diff': float(
                    reference_row['calinski_harabasz']
                    - row['calinski_harabasz']
                ),
                'calinski_harabasz_bootstrap_p_value': bootstrap_p_value(
                    diffs['calinski_harabasz_diff']
                ),
            }
        )

    table = pd.DataFrame(rows)
    table['silhouette_significant'] = (
        table['silhouette_bootstrap_p_value'] < alpha
    )

    return table


def build_stability_mannwhitney_table(
    partitions: pd.DataFrame,
    stability_iterations: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    """Compare stability-score distributions against the top-ranked partition.

    Each representative run was resampled independently during the stability
    stage (500 subsampling iterations per run), so the per-iteration
    stability scores of two different runs form independent samples. A
    Mann-Whitney U test is therefore appropriate to test whether the
    top-ranked partition reaches a significantly higher stability score than
    a competing partition.
    """
    reference_row = partitions.iloc[0]
    reference_values = stability_iterations.loc[
        stability_iterations['run_id'] == reference_row['representative_run_id'],
        'stability_score',
    ].dropna().to_numpy()

    rows: list[dict[str, Any]] = []

    for _, row in partitions.iloc[1:].iterrows():
        candidate_values = stability_iterations.loc[
            stability_iterations['run_id'] == row['representative_run_id'],
            'stability_score',
        ].dropna().to_numpy()

        statistic, p_value = mannwhitneyu(
            reference_values,
            candidate_values,
            alternative='two-sided',
        )

        n_reference = len(reference_values)
        n_candidate = len(candidate_values)
        rank_biserial = 1.0 - (2.0 * statistic) / (n_reference * n_candidate)

        rows.append(
            {
                'reference_partition_id': reference_row['partition_id'],
                'reference_run_id': reference_row['representative_run_id'],
                'reference_median_stability': float(
                    np.median(reference_values)
                ),
                'candidate_partition_id': row['partition_id'],
                'candidate_run_id': row['representative_run_id'],
                'candidate_final_rank': int(row['final_partition_rank']),
                'candidate_median_stability': float(
                    np.median(candidate_values)
                ),
                'n_reference_iterations': n_reference,
                'n_candidate_iterations': n_candidate,
                'mannwhitney_u_statistic': float(statistic),
                'rank_biserial_effect_size': float(rank_biserial),
                'mannwhitney_p_value': float(p_value),
            }
        )

    table = pd.DataFrame(rows)
    table['mannwhitney_p_value_fdr'] = apply_benjamini_hochberg(
        table['mannwhitney_p_value'].to_numpy()
    )
    table['stability_significant'] = table['mannwhitney_p_value_fdr'] < alpha

    return table


def build_silhouette_wilcoxon_table(
    values: np.ndarray,
    partitions: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    """Compare per-year silhouette widths with a paired Wilcoxon signed-rank test."""
    reference_row = partitions.iloc[0]
    reference_labels = parse_canonical_labels(reference_row['canonical_labels'])
    reference_widths = silhouette_samples(values, reference_labels)

    rows: list[dict[str, Any]] = []

    for _, row in partitions.iloc[1:].iterrows():
        candidate_labels = parse_canonical_labels(row['canonical_labels'])
        candidate_widths = silhouette_samples(values, candidate_labels)

        differences = reference_widths - candidate_widths

        if np.allclose(differences, 0.0):
            statistic, p_value = float('nan'), 1.0
        else:
            statistic, p_value = wilcoxon(
                reference_widths,
                candidate_widths,
                zero_method='wilcox',
                alternative='two-sided',
                method='auto',
            )

        rows.append(
            {
                'reference_partition_id': reference_row['partition_id'],
                'candidate_partition_id': row['partition_id'],
                'candidate_final_rank': int(row['final_partition_rank']),
                'mean_silhouette_width_diff': float(np.mean(differences)),
                'wilcoxon_statistic': (
                    float(statistic) if np.isfinite(statistic) else float('nan')
                ),
                'wilcoxon_p_value': float(p_value),
            }
        )

    table = pd.DataFrame(rows)
    table['wilcoxon_p_value_fdr'] = apply_benjamini_hochberg(
        table['wilcoxon_p_value'].to_numpy()
    )
    table['wilcoxon_significant'] = table['wilcoxon_p_value_fdr'] < alpha

    return table


def build_ari_matrix(partitions: pd.DataFrame) -> pd.DataFrame:
    """Build a pairwise Adjusted Rand Index matrix among unique partitions."""
    partition_ids = partitions['partition_id'].tolist()
    labels_by_partition = {
        partition_id: parse_canonical_labels(canonical_labels)
        for partition_id, canonical_labels in zip(
            partitions['partition_id'],
            partitions['canonical_labels'],
        )
    }

    matrix = pd.DataFrame(
        1.0,
        index=partition_ids,
        columns=partition_ids,
    )

    for id_a, id_b in combinations(partition_ids, 2):
        score = float(
            adjusted_rand_score(
                labels_by_partition[id_a],
                labels_by_partition[id_b],
            )
        )
        matrix.loc[id_a, id_b] = score
        matrix.loc[id_b, id_a] = score

    matrix.index.name = 'partition_id'

    return matrix.reset_index()


def classify_comparison(row: pd.Series) -> str:
    """Classify a candidate partition relative to the top-ranked partition."""
    silhouette_significant = bool(row['silhouette_significant'])
    stability_significant = bool(row['stability_significant'])

    if silhouette_significant and stability_significant:
        return 'diferencia_significativa'

    if not silhouette_significant and not stability_significant:
        return 'sin_diferencia_significativa'

    return 'resultado_mixto'


def build_significance_summary(
    partitions: pd.DataFrame,
    bootstrap_table: pd.DataFrame,
    stability_table: pd.DataFrame,
    ari_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Combine bootstrap, stability and agreement results into one verdict table."""
    reference_partition_id = str(partitions.iloc[0]['partition_id'])
    ari_lookup = ari_matrix.set_index('partition_id')[reference_partition_id]

    merged = bootstrap_table.merge(
        stability_table[
            [
                'candidate_partition_id',
                'mannwhitney_p_value',
                'mannwhitney_p_value_fdr',
                'stability_significant',
            ]
        ],
        on='candidate_partition_id',
        how='left',
    )

    merged['adjusted_rand_index_vs_top'] = merged['candidate_partition_id'].map(
        ari_lookup
    )
    merged['verdict'] = merged.apply(classify_comparison, axis=1)

    return merged.sort_values('candidate_final_rank').reset_index(drop=True)


def get_statistical_output_paths(
    experiments_config: dict[str, Any],
) -> dict[str, Path]:
    """Return output paths for statistical comparison tables."""
    output_dir = get_table_output_dir(experiments_config)

    return {
        'bootstrap_metrics': output_dir / 'statistical_bootstrap_metrics.csv',
        'stability_mannwhitney': (
            output_dir / 'statistical_stability_mannwhitney.csv'
        ),
        'silhouette_wilcoxon': output_dir / 'statistical_silhouette_wilcoxon.csv',
        'ari_matrix': output_dir / 'statistical_ari_matrix.csv',
        'summary': output_dir / 'statistical_significance_summary.csv',
    }


def save_statistical_outputs(
    bootstrap_table: pd.DataFrame,
    stability_table: pd.DataFrame,
    wilcoxon_table: pd.DataFrame,
    ari_matrix: pd.DataFrame,
    summary: pd.DataFrame,
    experiments_config: dict[str, Any],
) -> dict[str, Path]:
    """Save statistical comparison outputs."""
    paths = get_statistical_output_paths(experiments_config)
    paths['bootstrap_metrics'].parent.mkdir(parents=True, exist_ok=True)

    bootstrap_table.to_csv(paths['bootstrap_metrics'], index=False, encoding='utf-8')
    stability_table.to_csv(
        paths['stability_mannwhitney'], index=False, encoding='utf-8'
    )
    wilcoxon_table.to_csv(
        paths['silhouette_wilcoxon'], index=False, encoding='utf-8'
    )
    ari_matrix.to_csv(paths['ari_matrix'], index=False, encoding='utf-8')
    summary.to_csv(paths['summary'], index=False, encoding='utf-8')

    return paths


def run_statistical_tests(
    pipeline_config: dict[str, Any],
    experiments_config: dict[str, Any],
) -> pd.DataFrame:
    """Run statistical comparisons among the final ranked partitions."""
    n_bootstrap, alpha = get_statistical_settings(experiments_config)
    random_state = get_project_random_state(pipeline_config)

    values = load_annual_matrix(pipeline_config, experiments_config)
    partitions = load_final_partitions(experiments_config)
    stability_iterations = load_stability_iterations(experiments_config)

    bootstrap_table = build_bootstrap_comparison_table(
        values=values,
        partitions=partitions,
        random_state=random_state,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
    )
    stability_table = build_stability_mannwhitney_table(
        partitions=partitions,
        stability_iterations=stability_iterations,
        alpha=alpha,
    )
    wilcoxon_table = build_silhouette_wilcoxon_table(
        values=values,
        partitions=partitions,
        alpha=alpha,
    )
    ari_matrix = build_ari_matrix(partitions)
    summary = build_significance_summary(
        partitions=partitions,
        bootstrap_table=bootstrap_table,
        stability_table=stability_table,
        ari_matrix=ari_matrix,
    )

    paths = save_statistical_outputs(
        bootstrap_table=bootstrap_table,
        stability_table=stability_table,
        wilcoxon_table=wilcoxon_table,
        ari_matrix=ari_matrix,
        summary=summary,
        experiments_config=experiments_config,
    )

    top_partition_id = partitions.iloc[0]['partition_id']
    top_run_id = partitions.iloc[0]['representative_run_id']

    print('[statistical_tests] Comparación estadística entre agrupamientos')
    print(f'[statistical_tests] Partición de referencia: {top_partition_id}')
    print(f'[statistical_tests] Run representante: {top_run_id}')
    print(f'[statistical_tests] Iteraciones bootstrap: {n_bootstrap}')
    print(f'[statistical_tests] Nivel de significancia: {alpha}')
    print(f"[statistical_tests] Tabla bootstrap: {paths['bootstrap_metrics']}")
    print(
        '[statistical_tests] Tabla Mann-Whitney estabilidad: '
        f"{paths['stability_mannwhitney']}"
    )
    print(f"[statistical_tests] Tabla Wilcoxon silhouette: {paths['silhouette_wilcoxon']}")
    print(f"[statistical_tests] Matriz ARI: {paths['ari_matrix']}")
    print(f"[statistical_tests] Resumen de veredictos: {paths['summary']}")

    print('[statistical_tests] Veredictos frente a la partición de referencia:')
    print(
        summary[
            [
                'candidate_partition_id',
                'candidate_final_rank',
                'silhouette_bootstrap_p_value',
                'mannwhitney_p_value_fdr',
                'adjusted_rand_index_vs_top',
                'verdict',
            ]
        ].round(4).to_string(index=False)
    )

    return summary


if __name__ == '__main__':
    pipeline_config = load_pipeline_config()
    experiments_config = load_experiments_config()

    run_statistical_tests(
        pipeline_config=pipeline_config,
        experiments_config=experiments_config,
    )
