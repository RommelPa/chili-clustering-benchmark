"""Exploratory paired bootstrap of fixed, previously selected clusterings.

No null-calibrated p-values or superiority/equivalence verdicts are produced.
Percentile intervals condition on valid replicates and assume exchangeable years;
selection, serial dependence, refitting and scaling uncertainty are not covered.
"""
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config_loader import load_experiments_config, load_pipeline_config
from src.experiment_matrix import get_feature_sets, get_scalers
from src.metrics import safe_internal_metrics
from src.partition_comparison import parse_canonical_labels
from src.partition_equivalence import canonicalize_labels
from src.stability import get_project_random_state

METRIC_SIGNS = {'silhouette': 1, 'davies_bouldin': -1, 'calinski_harabasz': 1}


def make_bootstrap_indices(n_items: int, n_bootstrap: int, seed: int) -> np.ndarray:
    if n_items < 3 or n_bootstrap < 2:
        raise ValueError('Need at least 3 years and 2 bootstrap replicates.')
    return np.random.default_rng(seed).integers(0, n_items, size=(n_bootstrap, n_items))


def compare_fixed_clusterings(
    values: np.ndarray,
    partitions: pd.DataFrame,
    indices: np.ndarray,
    confidence: float = 0.95,
    progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use identical bootstrap rows for every reference/candidate comparison.

    Positive differences favor the reference for all three metrics. A replicate
    is retained for a pair only when all three metrics are finite for both.
    Degenerate samples are recorded, never silently replaced by new draws.
    Noise is rejected: comparing scores on different subsets would not be paired.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Expected a finite two-dimensional feature matrix.')
    if not 0 < confidence < 1:
        raise ValueError('confidence must lie between 0 and 1.')
    if (indices.ndim != 2 or indices.shape[1] != len(values)
            or len(indices) < 2 or not np.issubdtype(indices.dtype, np.integer)
            or indices.min() < 0 or indices.max() >= len(values)):
        raise ValueError('Invalid bootstrap indices.')
    ordered = partitions.sort_values('final_partition_rank').reset_index(drop=True)
    if len(ordered) < 2 or ordered.partition_id.duplicated().any():
        raise ValueError('Need at least two distinct grouping identifiers.')
    labels = [parse_canonical_labels(v) for v in ordered.canonical_labels]
    if any(len(v) != len(values) or (v < 0).any() for v in labels):
        raise ValueError('All groupings must cover the same years without noise.')
    observed = [safe_internal_metrics(values, v) for v in labels]
    if any(not all(np.isfinite(list(v.values()))) for v in observed):
        raise ValueError('Observed groupings must have computable internal metrics.')
    metrics = list(METRIC_SIGNS)
    def evaluate(sample_labels: np.ndarray) -> np.ndarray:
        rows = [safe_internal_metrics(values[sample], sample_labels[sample])
                for sample in indices]
        return np.array([[row[metric] for metric in metrics] for row in rows])

    reference = evaluate(labels[0])
    summaries: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    alpha = (1 - confidence) / 2
    for position in range(1, len(ordered)):
        if progress:
            print(f'[bootstrap] Comparison {position}/{len(ordered) - 1}', flush=True)
        candidate = evaluate(labels[position])
        valid = np.isfinite(reference).all(axis=1) & np.isfinite(candidate).all(axis=1)
        differences = (reference - candidate) * np.array(list(METRIC_SIGNS.values()))
        differences[~valid] = np.nan
        detail = pd.DataFrame({
            'bootstrap_id': np.arange(1, len(indices) + 1),
            'reference_grouping_id': ordered.iloc[0].partition_id,
            'candidate_grouping_id': ordered.iloc[position].partition_id,
            'valid_pair': valid,
        })
        for column, metric in enumerate(metrics):
            diff = differences[valid, column]
            detail[f'{metric}_difference'] = differences[:, column]
            low, high = (np.quantile(diff, [alpha, 1 - alpha])
                         if len(diff) else (np.nan, np.nan))
            summaries.append({
                'reference_grouping_id': ordered.iloc[0].partition_id,
                'reference_run_id': ordered.iloc[0].representative_run_id,
                'candidate_grouping_id': ordered.iloc[position].partition_id,
                'candidate_run_id': ordered.iloc[position].representative_run_id,
                'candidate_final_rank': ordered.iloc[position].final_partition_rank,
                'metric': metric,
                'observed_difference': METRIC_SIGNS[metric] * (
                    observed[0][metric] - observed[position][metric]),
                'n_requested': len(indices), 'n_valid': int(valid.sum()),
                'n_invalid': int((~valid).sum()), 'valid_fraction': float(valid.mean()),
                'bootstrap_mean_difference': float(diff.mean()) if len(diff) else np.nan,
                'percentile_low': low, 'percentile_high': high,
                'confidence_level': confidence,
                'scope': 'exploratory_fixed_labels_pointwise_conditional_on_valid',
            })
        frames.append(detail)
    return pd.DataFrame(summaries), pd.concat(frames, ignore_index=True)


def run_bootstrap_comparison(
    pipeline_config: dict[str, Any], experiments_config: dict[str, Any],
) -> pd.DataFrame:
    settings = experiments_config['comparacion_bootstrap']
    feature_sets = get_feature_sets(experiments_config)
    scalers = get_scalers(experiments_config)
    if len(feature_sets) != 1 or len(scalers) != 1:
        raise ValueError('Bootstrap requires exactly one feature set and scaler.')
    name = next(iter(feature_sets))
    directory = Path(experiments_config['salidas']['tablas']['benchmark_runs']).parent
    matrix = pd.read_csv(directory / f'matrix_{name}_{scalers[0]}.csv').sort_values('AH_YEAR')
    partitions = pd.read_csv(directory / 'final_partition_ranking.csv')
    benchmark_labels = pd.read_csv(directory / 'benchmark_labels.csv')
    if matrix.AH_YEAR.duplicated().any():
        raise ValueError('Duplicate matrix years.')
    # Check row alignment against the original representative labels, not length alone.
    for _, row in partitions.iterrows():
        original = benchmark_labels[benchmark_labels.run_id == row.representative_run_id].sort_values('AH_YEAR')
        if (original.AH_YEAR.tolist() != matrix.AH_YEAR.tolist()
                or list(canonicalize_labels(original.cluster_label.astype(int).tolist()))
                != parse_canonical_labels(row.canonical_labels).tolist()):
            raise ValueError('Grouping labels and annual matrix are not aligned.')
    indices = make_bootstrap_indices(
        len(matrix), int(settings['n_replicas']), get_project_random_state(pipeline_config),
    )
    print(f'[bootstrap] Starting {len(indices)} paired replicates; fixed labels', flush=True)
    summary, iterations = compare_fixed_clusterings(
        matrix[feature_sets[name]].to_numpy(), partitions, indices,
        float(settings['nivel_intervalo']), progress=True,
    )
    counts = np.array([np.bincount(sample, minlength=len(matrix)) for sample in indices])
    samples = pd.DataFrame(counts, columns=matrix.AH.astype(str).tolist())
    samples.insert(0, 'bootstrap_id', np.arange(1, len(indices) + 1))
    summary.to_csv(directory / 'bootstrap_comparison.csv', index=False)
    iterations.to_csv(directory / 'bootstrap_iterations.csv', index=False)
    samples.to_csv(directory / 'bootstrap_samples.csv', index=False)
    print('[bootstrap] Exploratory fixed-label paired bootstrap; no significance verdicts')
    print(f'[bootstrap] Replicates: {len(indices)}; comparisons: {len(partitions) - 1}')
    print(summary.head(3).to_string(index=False))
    return summary


if __name__ == '__main__':
    run_bootstrap_comparison(load_pipeline_config(), load_experiments_config())
