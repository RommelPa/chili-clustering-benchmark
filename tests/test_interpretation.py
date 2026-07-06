import pandas as pd

from src.interpretation import (
    build_cluster_feature_differences,
    build_cluster_profiles,
    build_selected_partition_membership,
    get_feature_columns,
    save_interpretation_outputs,
    select_representative_partition,
)


def build_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'AH': ['AH-2012', 'AH-2013', 'AH-2014', 'AH-2015'],
            'AH_YEAR': [2012, 2013, 2014, 2015],
            'vol_min': [1.0, 2.0, 9.0, 10.0],
            'vol_amp': [5.0, 6.0, 1.0, 2.0],
        }
    )


def build_labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'run_id': ['spectral_002'] * 4,
            'AH': ['AH-2012', 'AH-2013', 'AH-2014', 'AH-2015'],
            'AH_YEAR': [2012, 2013, 2014, 2015],
            'cluster_label': [5, 5, 2, 2],
        }
    )


def build_final_partitions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'final_partition_rank': [1],
            'partition_id': ['partition_001'],
            'representative_run_id': ['spectral_002'],
            'representative_algorithm': ['spectral'],
            'final_score': [0.97],
        }
    )


def test_get_feature_columns_excludes_metadata():
    features = build_features()

    columns = get_feature_columns(features)

    assert columns == ['vol_min', 'vol_amp']


def test_select_representative_partition_returns_rank_one():
    selected = select_representative_partition(build_final_partitions())

    assert selected['partition_id'] == 'partition_001'
    assert selected['representative_run_id'] == 'spectral_002'


def test_build_selected_partition_membership_adds_canonical_clusters():
    membership = build_selected_partition_membership(
        features=build_features(),
        labels=build_labels(),
        final_partitions=build_final_partitions(),
    )

    assert membership['cluster_id'].tolist() == [
        'cluster_1',
        'cluster_1',
        'cluster_2',
        'cluster_2',
    ]
    assert membership['selected_partition_id'].iloc[0] == 'partition_001'


def test_build_cluster_profiles_summarizes_years_and_means():
    membership = build_selected_partition_membership(
        features=build_features(),
        labels=build_labels(),
        final_partitions=build_final_partitions(),
    )

    profiles = build_cluster_profiles(
        membership,
        feature_columns=['vol_min', 'vol_amp'],
    )

    first_cluster = profiles[profiles['cluster_id'] == 'cluster_1'].iloc[0]

    assert first_cluster['n_years'] == 2
    assert first_cluster['vol_min_mean'] == 1.5
    assert 'AH-2012' in first_cluster['years']


def test_build_cluster_feature_differences_compares_global_mean():
    membership = build_selected_partition_membership(
        features=build_features(),
        labels=build_labels(),
        final_partitions=build_final_partitions(),
    )

    differences = build_cluster_feature_differences(
        membership,
        feature_columns=['vol_min', 'vol_amp'],
    )

    row = differences[
        (differences['cluster_id'] == 'cluster_1')
        & (differences['feature'] == 'vol_min')
    ].iloc[0]

    assert row['cluster_mean'] == 1.5
    assert row['global_mean'] == 5.5
    assert row['direction'] == 'below_global_mean'


def test_save_interpretation_outputs_writes_csv_files(tmp_path):
    membership = build_selected_partition_membership(
        features=build_features(),
        labels=build_labels(),
        final_partitions=build_final_partitions(),
    )
    profiles = build_cluster_profiles(
        membership,
        feature_columns=['vol_min', 'vol_amp'],
    )
    differences = build_cluster_feature_differences(
        membership,
        feature_columns=['vol_min', 'vol_amp'],
    )
    config = {
        'salidas': {
            'tablas': {
                'benchmark_runs': str(tmp_path / 'benchmark_runs.csv')
            }
        }
    }

    membership_path, profiles_path, differences_path = (
        save_interpretation_outputs(
            membership=membership,
            profiles=profiles,
            differences=differences,
            experiments_config=config,
        )
    )

    assert membership_path.exists()
    assert profiles_path.exists()
    assert differences_path.exists()
