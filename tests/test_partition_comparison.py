import pandas as pd
import pytest
from src.partition_comparison import build_ari_matrix, build_comparison_table


def test_identical_relabelled_partitions_have_ari_one():
    x = pd.DataFrame({'partition_id': ['a', 'b'], 'canonical_labels': ['0|0|1|1', '3|3|7|7']})
    assert build_ari_matrix(x).loc[0, 'b'] == pytest.approx(1)


def test_comparison_pairs_only_matching_successful_subsamples():
    partitions = pd.DataFrame({
        'partition_id': ['a', 'b'], 'final_partition_rank': [1, 2],
        'representative_run_id': ['r1', 'r2'], 'canonical_labels': ['0|0|1|1', '0|1|0|1'],
        'silhouette': [.5, .3], 'davies_bouldin': [.4, .7], 'calinski_harabasz': [8., 6.],
    })
    iterations = pd.DataFrame({
        'run_id': ['r1', 'r1', 'r2', 'r2'], 'iteration': [1, 2, 1, 2],
        'sample_signature': ['0|1|2', '1|2|3', '0|1|2', '0|2|3'],
        'status': ['success'] * 4, 'stability_score': [.9, .5, .7, .8],
    })
    result = build_comparison_table(partitions, iterations)
    assert result.loc[0, 'n_common_valid_subsamples'] == 1
    assert result.loc[0, 'mean_stability_difference'] == pytest.approx(.2)
    assert result.loc[0, 'davies_bouldin_difference'] == pytest.approx(.3)
    assert not any('p_value' in c or 'significant' in c for c in result)


@pytest.mark.parametrize("difference,expected", [
    (0.2, (1., 0., 0.)),
    (0., (0., 1., 0.)),
    (1e-14, (0., 1., 0.)),
    (-0.2, (0., 0., 1.)),
])
def test_comparison_separates_better_equal_and_worse(difference, expected):
    partitions = pd.DataFrame({
        'partition_id': ['a', 'b'], 'final_partition_rank': [1, 2],
        'representative_run_id': ['r1', 'r2'],
        'canonical_labels': ['0|0|1|1', '0|1|0|1'],
        'silhouette': [.5, .3], 'davies_bouldin': [.4, .7],
        'calinski_harabasz': [8., 6.],
    })
    iterations = pd.DataFrame({
        'run_id': ['r1', 'r2'], 'iteration': [1, 1],
        'sample_signature': ['0|1|2'] * 2, 'status': ['success'] * 2,
        'stability_score': [.5 + difference, .5],
    })
    row = build_comparison_table(partitions, iterations).iloc[0]
    fractions = tuple(row[c] for c in [
        'fraction_reference_more_stable', 'fraction_equal_stability',
        'fraction_reference_less_stable',
    ])
    assert fractions == pytest.approx(expected)
    assert sum(fractions) == pytest.approx(1.)
