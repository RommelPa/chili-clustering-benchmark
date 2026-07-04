from pathlib import Path

import pandas as pd
import pytest

from src.config_loader import load_pipeline_config
from src.preprocessing import (
    assign_hydrological_year,
    assign_operational_season,
    calculate_daily_operational_aggregates,
    filter_study_period,
    get_columns_by_prefix,
    preprocess_daily_data,
)


def test_filter_study_period_keeps_configured_dates():
    config = load_pipeline_config()
    data = pd.DataFrame(
        {
            'FECHA': pd.to_datetime(
                ['2011-11-30', '2011-12-01', '2025-11-30', '2025-12-01']
            )
        }
    )

    filtered = filter_study_period(data, config)

    assert filtered['FECHA'].min() == pd.Timestamp('2011-12-01')
    assert filtered['FECHA'].max() == pd.Timestamp('2025-11-30')
    assert len(filtered) == 2


def test_assign_hydrological_year_labels_december_as_next_year():
    config = load_pipeline_config()
    data = pd.DataFrame(
        {
            'FECHA': pd.to_datetime(['2011-12-01', '2012-01-01', '2012-11-30'])
        }
    )

    result = assign_hydrological_year(data, config)

    assert result['AH'].tolist() == ['AH-2012', 'AH-2012', 'AH-2012']
    assert result['AH_YEAR'].tolist() == [2012, 2012, 2012]


def test_assign_operational_season_uses_configured_months():
    config = load_pipeline_config()
    data = pd.DataFrame(
        {
            'FECHA': pd.to_datetime(['2020-01-15', '2020-07-15'])
        }
    )

    result = assign_operational_season(data, config)

    assert result['TEMPORADA'].tolist() == ['avenida', 'estiaje']


def test_get_columns_by_prefix_returns_seven_reservoir_columns():
    config = load_pipeline_config()

    columns = get_columns_by_prefix(config, 'VOLUTIL')

    assert len(columns) == 7
    assert 'VOLUTILAB' in columns
    assert 'VOLUTILBA' in columns


def test_calculate_daily_operational_aggregates_on_synthetic_data():
    config = load_pipeline_config()

    row = {}
    for suffix in config['embalses']['sufijos']:
        row[f'VOLUTIL{suffix}'] = 1_000_000
        row[f'DESCARGA{suffix}'] = 2.0
        row[f'REBOSE{suffix}'] = 0.0

    data = pd.DataFrame([row])

    result = calculate_daily_operational_aggregates(data, config)

    assert result.loc[0, 'VOL_TOTAL_HM3'] == 7.0
    assert result.loc[0, 'DESC_TOTAL_M3S'] == 14.0
    assert result.loc[0, 'REBOSE_TOTAL_M3S'] == 0.0
    assert result.loc[0, 'DERRAME_FLAG'] == 0


def test_preprocess_daily_data_returns_expected_period_if_file_exists():
    config = load_pipeline_config()
    raw_path = Path(config['rutas']['datos_crudos'])

    if not raw_path.exists():
        pytest.skip('Raw Excel file is not available in this environment.')

    data = preprocess_daily_data(config)

    assert not data.empty
    assert data['FECHA'].min() == pd.Timestamp('2011-12-01')
    assert data['FECHA'].max() == pd.Timestamp('2025-11-30')
    assert data['AH'].min() == 'AH-2012'
    assert data['AH'].max() == 'AH-2025'
    assert set(data['TEMPORADA'].unique()) == {'avenida', 'estiaje'}


def test_preprocess_daily_data_writes_processed_csv_if_file_exists():
    config = load_pipeline_config()
    raw_path = Path(config['rutas']['datos_crudos'])

    if not raw_path.exists():
        pytest.skip('Raw Excel file is not available in this environment.')

    output_path = Path(config['rutas']['dataset_procesado'])
    data = preprocess_daily_data(config)

    assert output_path.exists()
    assert len(data) > 0
