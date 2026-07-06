from pathlib import Path

import pandas as pd

from src.reporting import (
    build_methodology_results_report,
    build_years_by_cluster,
    dataframe_to_markdown,
    get_report_output_dir,
    save_report,
    summarize_benchmark_outputs,
)


def build_benchmark() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'run_id': ['run_001', 'run_002', 'run_003'],
            'status': ['success', 'success', 'success'],
            'is_valid_partition': [True, True, False],
        }
    )


def build_final_runs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'run_id': ['run_001', 'run_002'],
            'final_score': [0.97, 0.80],
        }
    )


def build_final_partitions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'final_partition_rank': [1],
            'partition_id': ['partition_001'],
            'representative_run_id': ['spectral_002'],
            'representative_algorithm': ['spectral'],
            'final_score': [0.9698],
            'ranking_score': [0.9700],
            'stability_score': [0.9993],
            'n_clusters': [2],
            'n_equivalent_runs': [13],
            'n_supporting_algorithms': [5],
        }
    )


def build_membership() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'AH': ['AH-2012', 'AH-2013', 'AH-2014', 'AH-2015'],
            'cluster_id': [
                'cluster_1',
                'cluster_1',
                'cluster_2',
                'cluster_2',
            ],
        }
    )


def build_profiles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'cluster_id': ['cluster_1', 'cluster_2'],
            'n_years': [2, 2],
            'vol_min_mean': [0.38, 0.24],
            'inflow_estiaje_mean': [12.9, 10.3],
        }
    )


def build_differences() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'cluster_id': ['cluster_2', 'cluster_1'],
            'feature': ['vol_min', 'vol_min'],
            'cluster_mean': [0.24, 0.38],
            'global_mean': [0.34, 0.34],
            'difference': [-0.10, 0.04],
            'standardized_difference': [-1.14, 0.45],
            'direction': ['below_global_mean', 'above_global_mean'],
        }
    )


def test_get_report_output_dir_uses_fallback():
    config = {
        'salidas': {
            'tablas': {
                'benchmark_runs': 'outputs/tables/benchmark_runs.csv',
            }
        }
    }

    assert get_report_output_dir(config) == Path('outputs/reports')


def test_summarize_benchmark_outputs_counts_valid_runs():
    summary = summarize_benchmark_outputs(
        benchmark=build_benchmark(),
        final_runs=build_final_runs(),
        final_partitions=build_final_partitions(),
    )

    assert summary['n_total_configurations'] == 3
    assert summary['n_successful_configurations'] == 3
    assert summary['n_valid_partitions'] == 2
    assert summary['n_ranked_runs'] == 2
    assert summary['n_unique_partitions'] == 1


def test_dataframe_to_markdown_renders_table():
    data = pd.DataFrame(
        {
            'name': ['a'],
            'score': [0.12345],
        }
    )

    markdown = dataframe_to_markdown(data, digits=2)

    assert '| name | score |' in markdown
    assert '| a | 0.12 |' in markdown


def test_build_years_by_cluster_groups_years():
    years = build_years_by_cluster(build_membership())

    assert len(years) == 2
    assert years.iloc[0]['n_years'] == 2
    assert 'AH-2012' in years.iloc[0]['years']


def test_build_methodology_results_report_contains_top_partition():
    report = build_methodology_results_report(
        benchmark=build_benchmark(),
        final_runs=build_final_runs(),
        final_partitions=build_final_partitions(),
        membership=build_membership(),
        profiles=build_profiles(),
        differences=build_differences(),
    )

    assert '# Resumen metodológico' in report
    assert 'partition_001' in report
    assert 'spectral_002' in report
    assert 'mayor disponibilidad hídrica-operativa' in report


def test_save_report_writes_markdown_file(tmp_path):
    output_path = tmp_path / 'report.md'

    result = save_report('# Reporte\n', output_path)

    assert result.exists()
    assert result.read_text(encoding='utf-8') == '# Reporte\n'
