from pathlib import Path

import pandas as pd
import pytest

from src.config_loader import load_pipeline_config
from src.data_loader import (
    build_raw_expected_columns,
    build_schema,
    build_standard_expected_columns,
    load_raw_data,
)


def test_build_raw_expected_columns_has_40_columns():
    config = load_pipeline_config()
    columns = build_raw_expected_columns(config)

    assert len(columns) == 40
    assert 'FECHA' in columns
    assert 'INGCHINEGASA' in columns


def test_raw_expected_columns_include_file_exception():
    config = load_pipeline_config()
    columns = build_raw_expected_columns(config)

    assert 'EVAPORACIONDEI' in columns


def test_standard_expected_columns_rename_exception():
    config = load_pipeline_config()
    columns = build_standard_expected_columns(config)

    assert 'EVAPORACIONDE' in columns
    assert 'EVAPORACIONDEI' not in columns


def test_build_schema_has_expected_columns():
    config = load_pipeline_config()
    schema = build_schema(config)

    assert len(schema.columns) == 40
    assert 'FECHA' in schema.columns
    assert 'EVAPORACIONDE' in schema.columns


def test_load_raw_data_returns_dataframe_if_file_exists():
    config = load_pipeline_config()
    raw_path = Path(config['rutas']['datos_crudos'])

    if not raw_path.exists():
        pytest.skip('Raw Excel file is not available in this environment.')

    data = load_raw_data(config)

    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert data.shape[1] == 40


def test_load_raw_data_has_standardized_exception_if_file_exists():
    config = load_pipeline_config()
    raw_path = Path(config['rutas']['datos_crudos'])

    if not raw_path.exists():
        pytest.skip('Raw Excel file is not available in this environment.')

    data = load_raw_data(config)

    assert 'EVAPORACIONDE' in data.columns
    assert 'EVAPORACIONDEI' not in data.columns


def test_load_raw_data_has_valid_dates_if_file_exists():
    config = load_pipeline_config()
    raw_path = Path(config['rutas']['datos_crudos'])

    if not raw_path.exists():
        pytest.skip('Raw Excel file is not available in this environment.')

    data = load_raw_data(config)

    assert pd.api.types.is_datetime64_any_dtype(data['FECHA'])
    assert data['FECHA'].notna().all()
