from pathlib import Path

import pandas as pd

from src.data_summary import (
    build_data_processing_summary,
    calculate_max_abs_correlation,
    parse_cleaning_log,
    save_data_processing_summary,
)


def build_config(tmp_path: Path) -> dict:
    return {
        'rutas': {
            'dataset_procesado': str(tmp_path / 'dataset_diario.csv'),
            'features': str(tmp_path / 'features.csv'),
            'log_limpieza': str(tmp_path / 'log_limpieza.txt'),
            'carpeta_tablas': str(tmp_path / 'tables'),
        },
        'features': {
            'base': [
                {'codigo': 'vol_min'},
                {'codigo': 'vol_amp'},
                {'codigo': 'inflow_avenida'},
                {'codigo': 'inflow_estiaje'},
                {'codigo': 'dias_derrame'},
            ]
        },
    }


def build_daily_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'FECHA': [
                '2011-12-01',
                '2012-01-01',
                '2012-12-01',
                '2013-01-01',
            ],
            'AH': ['AH-2012', 'AH-2012', 'AH-2013', 'AH-2013'],
            'AH_YEAR': [2012, 2012, 2013, 2013],
            'TEMPORADA': ['avenida', 'avenida', 'avenida', 'avenida'],
            'DERRAME_FLAG': [0, 1, 0, 1],
            'VOL_TOTAL_REL': [0.4, 0.5, 0.3, 0.6],
        }
    )


def build_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'AH': ['AH-2012', 'AH-2013'],
            'AH_YEAR': [2012, 2013],
            'vol_min': [0.4, 0.3],
            'vol_amp': [0.1, 0.3],
            'inflow_avenida': [20.0, 15.0],
            'inflow_estiaje': [10.0, 8.0],
            'dias_derrame': [1, 1],
        }
    )


def build_correlations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'vol_min': [1.0, 0.5, -0.2],
            'vol_amp': [0.5, 1.0, 0.8],
            'inflow_avenida': [-0.2, 0.8, 1.0],
        },
        index=['vol_min', 'vol_amp', 'inflow_avenida'],
    )


def test_parse_cleaning_log_extracts_expected_metrics(tmp_path):
    log_path = tmp_path / 'log_limpieza.txt'
    log_path.write_text(
        '\n'.join(
            [
                'Log de carga de datos crudos',
                'Filas iniciales: 9871',
                'Filas descartadas por FECHA inválida: 0',
                'Filas finales: 9871',
                'Celdas numéricas perdidas por conversión: 3',
            ]
        ),
        encoding='utf-8',
    )

    parsed = parse_cleaning_log(log_path)

    assert parsed['initial_rows'] == '9871'
    assert parsed['raw_final_rows'] == '9871'
    assert parsed['lost_numeric_cells'] == '3'


def test_calculate_max_abs_correlation_uses_upper_triangle():
    value = calculate_max_abs_correlation(build_correlations())

    assert value == 0.8


def test_build_data_processing_summary_contains_core_metrics(tmp_path):
    summary = build_data_processing_summary(
        daily_data=build_daily_data(),
        features=build_features(),
        correlations=build_correlations(),
        config=build_config(tmp_path),
        cleaning_log={'initial_rows': '10'},
    )

    metrics = set(summary['metric'])

    assert 'initial_rows' in metrics
    assert 'daily_rows' in metrics
    assert 'hydrological_year_count' in metrics
    assert 'feature_count' in metrics
    assert 'max_abs_pairwise_correlation' in metrics


def test_build_data_processing_summary_formats_feature_names(tmp_path):
    summary = build_data_processing_summary(
        daily_data=build_daily_data(),
        features=build_features(),
        correlations=build_correlations(),
        config=build_config(tmp_path),
    )

    feature_names = summary.loc[
        summary['metric'] == 'feature_names',
        'value',
    ].iloc[0]

    assert 'vol_min' in feature_names
    assert 'dias_derrame' in feature_names


def test_save_data_processing_summary_writes_csv(tmp_path):
    config = build_config(tmp_path)
    summary = build_data_processing_summary(
        daily_data=build_daily_data(),
        features=build_features(),
        correlations=build_correlations(),
        config=config,
    )

    output_path = save_data_processing_summary(summary, config)

    assert output_path.exists()

def test_parse_cleaning_log_handles_literal_newline_sequences(tmp_path):
    log_path = tmp_path / 'log_limpieza.txt'
    log_path.write_text(
        (
            'Log de carga de datos crudos\\n'
            'Filas iniciales: 9994\\n'
            'Filas descartadas por FECHA inválida: 123\\n'
            'Filas finales: 9871\\n'
            'Celdas numéricas perdidas por conversión: 244'
        ),
        encoding='utf-8',
    )

    parsed = parse_cleaning_log(log_path)

    assert parsed['initial_rows'] == '9994'
    assert parsed['invalid_date_rows_removed'] == '123'
    assert parsed['raw_final_rows'] == '9871'
    assert parsed['lost_numeric_cells'] == '244'
