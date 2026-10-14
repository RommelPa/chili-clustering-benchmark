"""Descriptive comparisons conditional on the observed annual dataset.

Repeated subsamples are perturbations of the same record, not independent
hydrological observations. No population significance or equivalence is inferred.
"""
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def parse_canonical_labels(labels: str) -> np.ndarray:
    return np.array([int(v) for v in str(labels).split('|')], dtype=int)


def build_ari_matrix(partitions: pd.DataFrame) -> pd.DataFrame:
    ids = partitions['partition_id'].tolist()
    labels = [parse_canonical_labels(v) for v in partitions['canonical_labels']]
    if len({len(v) for v in labels}) != 1:
        raise ValueError('Partitions must cover the same observations.')
    matrix = pd.DataFrame(np.eye(len(ids)), index=ids, columns=ids)
    for i, left in enumerate(labels):
        for j in range(i + 1, len(labels)):
            matrix.iloc[i, j] = matrix.iloc[j, i] = adjusted_rand_score(left, labels[j])
    matrix.index.name = 'partition_id'
    return matrix.reset_index()


def build_comparison_table(
    partitions: pd.DataFrame, iterations: pd.DataFrame,
) -> pd.DataFrame:
    """Compare scores on common perturbations without treating them as new years."""
    partitions = partitions.sort_values('final_partition_rank')
    reference = partitions.iloc[0]
    ref = iterations[iterations.run_id == reference.representative_run_id]
    ref = ref[ref.status == 'success'][['iteration', 'sample_signature', 'stability_score']]
    rows = []
    for _, candidate in partitions.iloc[1:].iterrows():
        other = iterations[iterations.run_id == candidate.representative_run_id]
        other = other[other.status == 'success']
        paired = ref.merge(other, on=['iteration', 'sample_signature'], suffixes=('_ref', '_other'), validate='one_to_one')
        differences = (paired.stability_score_ref - paired.stability_score_other).dropna()
        if differences.empty:
            raise ValueError('No common valid stability subsamples.')
        row = {
            'reference_partition_id': reference.partition_id,
            'candidate_partition_id': candidate.partition_id,
            'candidate_final_rank': candidate.final_partition_rank,
            'adjusted_rand_index': adjusted_rand_score(
                parse_canonical_labels(reference.canonical_labels),
                parse_canonical_labels(candidate.canonical_labels),
            ),
            'n_common_valid_subsamples': len(differences),
            'mean_stability_difference': differences.mean(),
            'stability_difference_q025': differences.quantile(0.025),
            'stability_difference_q975': differences.quantile(0.975),
            'fraction_reference_more_stable': (differences > 1e-12).mean(),
            'fraction_equal_stability': (differences.abs() <= 1e-12).mean(),
            'fraction_reference_less_stable': (differences < -1e-12).mean(),
        }
        for metric, sign in [('silhouette', 1), ('davies_bouldin', -1), ('calinski_harabasz', 1)]:
            row[f'{metric}_difference'] = sign * (reference[metric] - candidate[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def run_partition_comparison(
    pipeline_config: dict[str, Any], experiments_config: dict[str, Any],
) -> pd.DataFrame:
    directory = Path(experiments_config['salidas']['tablas']['benchmark_runs']).parent
    partitions = pd.read_csv(directory / 'final_partition_ranking.csv')
    iterations = pd.read_csv(directory / 'stability_iterations.csv')
    matrix = build_ari_matrix(partitions)
    summary = build_comparison_table(partitions, iterations)
    matrix.to_csv(directory / 'partition_ari_matrix.csv', index=False)
    summary.to_csv(directory / 'partition_comparison.csv', index=False)
    # These files belonged to the superseded inference stage. Remove only these
    # known outputs so old conclusions cannot be mistaken for current results.
    for name in ['statistical_bootstrap_metrics.csv', 'statistical_stability_mannwhitney.csv',
                 'statistical_silhouette_wilcoxon.csv', 'statistical_ari_matrix.csv',
                 'statistical_significance_summary.csv']:
        (directory / name).unlink(missing_ok=True)
    print('[partition_comparison] Descriptive comparison; no significance verdicts')
    return summary


if __name__ == '__main__':
    from src.config_loader import load_experiments_config, load_pipeline_config

    comparison = run_partition_comparison(
        load_pipeline_config(), load_experiments_config(),
    )
    print(comparison.to_string(index=False))
