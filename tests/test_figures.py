from pathlib import Path

import pandas as pd

from src.figures import (
    get_figure_output_dir,
    plot_cluster_difference_profile,
    plot_final_partition_scores,
    plot_year_typology,
    prepare_cluster_difference_matrix,
    prepare_partition_score_data,
    prepare_year_typology_data,
)


def build_final_partitions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'final_partition_rank': [1, 2],
            'partition_id': ['partition_001', 'partition_002'],
            'final_score': [0.97, 0.69],
            'stability_score': [0.99, 0.82],
            'ranking_score': [0.97, 0.65],
            'n_clusters': [2, 3],
        }
    )


def build_differences() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'cluster_id': [
                'cluster_1',
                'cluster_2',
                'cluster_1',
                'cluster_2',
            ],
            'feature': [
                'vol_min',
                'vol_min',
                'inflow_estiaje',
                'inflow_estiaje',
            ],
            'standardized_difference': [0.45, -1.14, 0.56, -1.39],
        }
    )


def build_membership() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'AH': ['AH-2012', 'AH-2013', 'AH-2014', 'AH-2015'],
            'AH_YEAR': [2012, 2013, 2014, 2015],
            'cluster_id': [
                'cluster_1',
                'cluster_1',
                'cluster_2',
                'cluster_2',
            ],
        }
    )


def test_get_figure_output_dir_uses_fallback():
    config = {
        'salidas': {
            'tablas': {
                'benchmark_runs': 'outputs/tables/benchmark_runs.csv',
            }
        }
    }

    assert get_figure_output_dir(config) == Path('outputs/figures')


def test_prepare_partition_score_data_keeps_top_ranked_rows():
    data = prepare_partition_score_data(build_final_partitions(), top_n=1)

    assert len(data) == 1
    assert data.iloc[0]['partition_id'] == 'partition_001'


def test_prepare_cluster_difference_matrix_orders_by_absolute_difference():
    matrix = prepare_cluster_difference_matrix(build_differences())

    assert matrix.index[0] == 'inflow_estiaje'
    assert 'cluster_1' in matrix.columns
    assert 'cluster_2' in matrix.columns


def test_prepare_year_typology_data_adds_cluster_position():
    data = prepare_year_typology_data(build_membership())

    assert 'cluster_position' in data.columns
    assert data['AH_YEAR'].tolist() == [2012, 2013, 2014, 2015]


def test_plot_final_partition_scores_writes_png(tmp_path):
    output_path = tmp_path / 'final_partition_scores.png'

    plot_final_partition_scores(
        final_partitions=build_final_partitions(),
        output_path=output_path,
    )

    assert output_path.exists()


def test_plot_cluster_difference_profile_writes_png(tmp_path):
    output_path = tmp_path / 'cluster_difference_profile.png'

    plot_cluster_difference_profile(
        differences=build_differences(),
        output_path=output_path,
    )

    assert output_path.exists()


def test_plot_year_typology_writes_png(tmp_path):
    output_path = tmp_path / 'year_typology.png'

    plot_year_typology(
        membership=build_membership(),
        output_path=output_path,
    )

    assert output_path.exists()

