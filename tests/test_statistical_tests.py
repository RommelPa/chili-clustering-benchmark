import numpy as np
import pandas as pd
import pytest

from src.statistical_tests import (
    apply_benjamini_hochberg,
    bootstrap_confidence_interval,
    bootstrap_metric_differences,
    bootstrap_p_value,
    build_ari_matrix,
    build_bootstrap_comparison_table,
    build_significance_summary,
    build_silhouette_wilcoxon_table,
    build_stability_mannwhitney_table,
    classify_comparison,
    get_statistical_settings,
    parse_canonical_labels,
)


def build_separable_values() -> np.ndarray:
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=-5.0, scale=0.2, size=(10, 2))
    cluster_b = rng.normal(loc=5.0, scale=0.2, size=(4, 2))

    return np.vstack([cluster_a, cluster_b])


def build_partitions_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'final_partition_rank': [1, 2],
            'partition_id': ['partition_001', 'partition_003'],
            'representative_run_id': ['spectral_002', 'agglomerative_004'],
            'silhouette': [0.4459, 0.4054],
            'davies_bouldin': [0.7515, 0.7523],
            'calinski_harabasz': [12.5584, 11.6066],
            'canonical_labels': [
                '0|0|1|1|1|0|0|0|0|0|0|1|0|0',
                '0|0|1|1|1|2|0|0|0|0|0|1|2|0',
            ],
        }
    )


def test_parse_canonical_labels_returns_int_array():
    labels = parse_canonical_labels('0|0|1|1|1|0|0|0|0|0|0|1|0|0')

    assert labels.dtype == np.dtype(int)
    assert labels.tolist() == [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]


def test_get_statistical_settings_uses_defaults_when_missing():
    n_bootstrap, alpha = get_statistical_settings({})

    assert n_bootstrap == 10000
    assert alpha == pytest.approx(0.05)


def test_get_statistical_settings_reads_configured_values():
    config = {
        'pruebas_estadisticas': {
            'n_bootstrap': 500,
            'nivel_significancia': 0.1,
        }
    }

    n_bootstrap, alpha = get_statistical_settings(config)

    assert n_bootstrap == 500
    assert alpha == pytest.approx(0.1)


def test_bootstrap_p_value_is_near_zero_for_consistently_positive_diffs():
    diffs = np.full(200, 0.05)

    assert bootstrap_p_value(diffs) == pytest.approx(0.0)


def test_bootstrap_p_value_is_one_for_symmetric_diffs_around_zero():
    diffs = np.array([-1.0, -1.0, 1.0, 1.0])

    assert bootstrap_p_value(diffs) == pytest.approx(1.0)


def test_bootstrap_p_value_returns_nan_for_empty_array():
    assert np.isnan(bootstrap_p_value(np.array([])))


def test_bootstrap_confidence_interval_brackets_percentiles():
    diffs = np.linspace(-1.0, 1.0, 1001)

    lower, upper = bootstrap_confidence_interval(diffs, confidence=0.95)

    assert lower == pytest.approx(-0.95, abs=0.01)
    assert upper == pytest.approx(0.95, abs=0.01)


def test_apply_benjamini_hochberg_preserves_order_of_smallest_p_value():
    p_values = np.array([0.20, 0.01, 0.30, 0.04])

    adjusted = apply_benjamini_hochberg(p_values)

    assert adjusted[np.argmin(p_values)] == pytest.approx(min(adjusted))
    assert np.all(adjusted >= p_values)


def test_apply_benjamini_hochberg_handles_empty_input():
    assert apply_benjamini_hochberg(np.array([])).size == 0


def test_bootstrap_metric_differences_favors_better_separated_partition():
    values = build_separable_values()
    true_labels = np.array([0] * 10 + [1] * 4)
    shuffled_labels = np.array([0, 1] * 7)

    diffs = bootstrap_metric_differences(
        values=values,
        labels_a=true_labels,
        labels_b=shuffled_labels,
        n_bootstrap=200,
        random_state=42,
    )

    assert diffs['silhouette_diff'].size > 0
    assert np.mean(diffs['silhouette_diff']) > 0
    assert bootstrap_p_value(diffs['silhouette_diff']) < 0.05


def test_bootstrap_metric_differences_are_near_zero_for_identical_partitions():
    values = build_separable_values()
    labels = np.array([0] * 10 + [1] * 4)

    diffs = bootstrap_metric_differences(
        values=values,
        labels_a=labels,
        labels_b=labels,
        n_bootstrap=100,
        random_state=1,
    )

    assert np.allclose(diffs['silhouette_diff'], 0.0)


def test_build_bootstrap_comparison_table_has_one_row_per_candidate():
    values = build_separable_values()
    partitions = build_partitions_frame()

    table = build_bootstrap_comparison_table(
        values=values,
        partitions=partitions,
        random_state=42,
        n_bootstrap=200,
        alpha=0.05,
    )

    assert len(table) == 1
    assert table.loc[0, 'candidate_partition_id'] == 'partition_003'
    assert 'silhouette_significant' in table.columns


def test_build_silhouette_wilcoxon_table_returns_expected_columns():
    values = build_separable_values()
    partitions = build_partitions_frame()

    table = build_silhouette_wilcoxon_table(
        values=values, partitions=partitions, alpha=0.05,
    )

    assert len(table) == 1
    assert 'wilcoxon_p_value' in table.columns
    assert 'wilcoxon_p_value_fdr' in table.columns


def test_build_ari_matrix_has_diagonal_of_ones():
    partitions = build_partitions_frame()

    matrix = build_ari_matrix(partitions)

    assert matrix.loc[0, 'partition_001'] == pytest.approx(1.0)
    assert matrix.loc[1, 'partition_003'] == pytest.approx(1.0)
    assert 0.0 <= matrix.loc[0, 'partition_003'] <= 1.0


def test_build_stability_mannwhitney_table_flags_lower_stability_run():
    partitions = build_partitions_frame()
    rng = np.random.default_rng(0)

    iterations = pd.DataFrame(
        {
            'run_id': ['spectral_002'] * 300 + ['agglomerative_004'] * 300,
            'stability_score': np.concatenate(
                [
                    rng.normal(0.999, 0.001, 300),
                    rng.normal(0.955, 0.02, 300),
                ]
            ),
        }
    )

    table = build_stability_mannwhitney_table(
        partitions=partitions,
        stability_iterations=iterations,
        alpha=0.05,
    )

    assert len(table) == 1
    assert table.loc[0, 'mannwhitney_p_value_fdr'] < 0.05
    assert bool(table.loc[0, 'stability_significant']) is True


def test_classify_comparison_labels_combinations():
    both_significant = pd.Series(
        {'silhouette_significant': True, 'stability_significant': True}
    )
    neither_significant = pd.Series(
        {'silhouette_significant': False, 'stability_significant': False}
    )
    mixed = pd.Series(
        {'silhouette_significant': True, 'stability_significant': False}
    )

    assert classify_comparison(both_significant) == 'diferencia_significativa'
    assert (
        classify_comparison(neither_significant)
        == 'sin_diferencia_significativa'
    )
    assert classify_comparison(mixed) == 'resultado_mixto'


def test_build_significance_summary_merges_all_sources():
    values = build_separable_values()
    partitions = build_partitions_frame()

    bootstrap_table = build_bootstrap_comparison_table(
        values=values,
        partitions=partitions,
        random_state=42,
        n_bootstrap=100,
        alpha=0.05,
    )

    rng = np.random.default_rng(0)
    iterations = pd.DataFrame(
        {
            'run_id': ['spectral_002'] * 100 + ['agglomerative_004'] * 100,
            'stability_score': np.concatenate(
                [
                    rng.normal(0.999, 0.001, 100),
                    rng.normal(0.955, 0.02, 100),
                ]
            ),
        }
    )
    stability_table = build_stability_mannwhitney_table(
        partitions=partitions,
        stability_iterations=iterations,
        alpha=0.05,
    )
    ari_matrix = build_ari_matrix(partitions)

    summary = build_significance_summary(
        partitions=partitions,
        bootstrap_table=bootstrap_table,
        stability_table=stability_table,
        ari_matrix=ari_matrix,
    )

    assert len(summary) == 1
    assert 'verdict' in summary.columns
    assert 'adjusted_rand_index_vs_top' in summary.columns
