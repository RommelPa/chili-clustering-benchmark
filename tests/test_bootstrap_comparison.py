import numpy as np
import pandas as pd
import pytest

from src.bootstrap_comparison import compare_fixed_clusterings, make_bootstrap_indices
from src.metrics import safe_internal_metrics


def example():
    values = np.array([[0., 0.], [.1, .2], [.2, 0.], [4., 4.], [4.1, 4.], [4.2, 4.1]])
    partitions = pd.DataFrame({
        'partition_id': ['a', 'b'], 'final_partition_rank': [1, 2],
        'representative_run_id': ['run_a', 'run_b'],
        'canonical_labels': ['0|0|0|1|1|1', '0|0|1|1|1|1'],
    })
    return values, partitions


def test_identical_relabelled_groupings_have_zero_differences_and_intervals():
    values, partitions = example()
    partitions.loc[1, 'canonical_labels'] = '8|8|8|9|9|9'
    indices = make_bootstrap_indices(6, 30, 42)
    summary, detail = compare_fixed_clusterings(values, partitions, indices)
    assert len(summary) == 3
    assert len(detail) == 30
    for column in ['observed_difference', 'percentile_low', 'percentile_high']:
        assert np.allclose(summary[column], 0)
    assert not any('p_value' in col or 'significant' in col for col in summary)


def test_same_draws_are_reproducible_and_shared_across_candidates():
    values, partitions = example()
    partitions.loc[2] = ['c', 3, 'run_c', partitions.loc[1, 'canonical_labels']]
    indices = make_bootstrap_indices(6, 25, 42)
    assert np.array_equal(indices, make_bootstrap_indices(6, 25, 42))
    summary, detail = compare_fixed_clusterings(values, partitions, indices)
    a = detail[detail.candidate_grouping_id == 'b'].reset_index(drop=True)
    b = detail[detail.candidate_grouping_id == 'c'].reset_index(drop=True)
    assert a.drop(columns='candidate_grouping_id').equals(b.drop(columns='candidate_grouping_id'))
    assert summary.n_requested.eq(25).all()


def test_degenerate_draw_is_counted_not_replaced_and_metric_directions_are_correct():
    values, partitions = example()
    indices = np.array([[0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5]])
    summary, detail = compare_fixed_clusterings(values, partitions, indices)
    assert summary.n_valid.eq(1).all()
    assert summary.n_invalid.eq(1).all()
    assert summary.valid_fraction.eq(.5).all()
    assert detail.valid_pair.tolist() == [False, True]
    assert detail.filter(like='_difference').iloc[0].isna().all()
    left = safe_internal_metrics(values, np.array([0, 0, 0, 1, 1, 1]))
    right = safe_internal_metrics(values, np.array([0, 0, 1, 1, 1, 1]))
    for _, row in summary.iterrows():
        sign = -1 if row.metric == 'davies_bouldin' else 1
        assert row.observed_difference == pytest.approx(sign * (left[row.metric] - right[row.metric]))
        assert row.percentile_low == pytest.approx(row.observed_difference)


def test_no_valid_draws_produces_missing_intervals():
    values, partitions = example()
    summary, _ = compare_fixed_clusterings(values, partitions, np.zeros((2, 6), dtype=int))
    assert summary.n_valid.eq(0).all()
    assert summary.percentile_low.isna().all()
    assert summary.percentile_high.isna().all()


def test_noise_and_misaligned_labels_are_rejected():
    values, partitions = example()
    for labels in ['0|0|0|1|1|-1', '0|0|1']:
        partitions.loc[1, 'canonical_labels'] = labels
        with pytest.raises(ValueError, match='same years without noise'):
            compare_fixed_clusterings(values, partitions, make_bootstrap_indices(6, 2, 42))


def test_cli_stage_writes_auditable_year_counts_and_checks_alignment(tmp_path):
    from src.bootstrap_comparison import run_bootstrap_comparison

    values, partitions = example()
    years = list(range(2012, 2018))
    matrix = pd.DataFrame(values, columns=['x', 'y'])
    matrix['AH_YEAR'] = years
    matrix['AH'] = [f'AH-{year}' for year in years]
    matrix.to_csv(tmp_path / 'matrix_base_zscore.csv', index=False)
    partitions.to_csv(tmp_path / 'final_partition_ranking.csv', index=False)
    labels = pd.DataFrame({
        'run_id': ['run_a'] * 6 + ['run_b'] * 6,
        'AH_YEAR': years * 2, 'cluster_label': [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
    })
    labels.to_csv(tmp_path / 'benchmark_labels.csv', index=False)
    config = {
        'feature_sets': {'base': ['x', 'y']}, 'scalers': ['zscore'],
        'salidas': {'tablas': {'benchmark_runs': str(tmp_path / 'benchmark_runs.csv')}},
        'comparacion_bootstrap': {'n_replicas': 10, 'nivel_intervalo': .95},
    }
    config['preprocesamiento_experimental'] = {
        'feature_sets': {'base': {'variables': ['x', 'y']}}, 'scalers': ['zscore'],
    }
    summary = run_bootstrap_comparison({'proyecto': {'semilla_aleatoria': 42}}, config)
    assert len(summary) == 3
    samples = pd.read_csv(tmp_path / 'bootstrap_samples.csv')
    assert samples.drop(columns='bootstrap_id').sum(axis=1).eq(6).all()
    assert (tmp_path / 'bootstrap_iterations.csv').exists()
    labels.loc[0, 'AH_YEAR'] = 1999
    labels.to_csv(tmp_path / 'benchmark_labels.csv', index=False)
    with pytest.raises(ValueError, match='not aligned'):
        run_bootstrap_comparison({'proyecto': {'semilla_aleatoria': 42}}, config)
