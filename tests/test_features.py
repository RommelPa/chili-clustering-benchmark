from pathlib import Path

import pandas as pd
import pytest

from src.config_loader import load_pipeline_config
from src.features import (
    calculate_feature_correlations,
    compute_annual_feature_matrix,
    generate_annual_features,
    get_feature_columns,
    validate_daily_dataset,
    validate_feature_matrix,
)


def build_synthetic_daily_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'FECHA': pd.to_datetime(
                [
                    '2011-12-01',
                    '2012-01-01',
                    '2012-07-01',
                    '2012-11-30',
                    '2012-12-01',
                    '2013-01-01',
                    '2013-07-01',
                    '2013-11-30',
                ]
            ),
            'AH_YEAR': [2012, 2012, 2012, 2012, 2013, 2013, 2013, 2013],
            'AH': [
                'AH-2012',
                'AH-2012',
                'AH-2012',
                'AH-2012',
                'AH-2013',
                'AH-2013',
                'AH-2013',
                'AH-2013',
            ],
            'TEMPORADA': [
                'avenida',
                'avenida',
                'estiaje',
                'estiaje',
                'avenida',
                'avenida',
                'estiaje',
                'estiaje',
            ],
            'INGCHINEGASA': [10.0, 20.0, 4.0, 6.0, 30.0, 40.0, 8.0, 12.0],
            'VOL_TOTAL_REL': [0.20, 0.50, 0.30, 0.25, 0.10, 0.60, 0.20, 0.15],
            'DERRAME_FLAG': [0, 1, 0, 0, 1, 1, 0, 0],
        }
    )


def test_get_feature_columns_returns_configured_features():
    config = load_pipeline_config()

    columns = get_feature_columns(config)

    assert columns == [
        'vol_min',
        'vol_amp',
        'inflow_avenida',
        'inflow_estiaje',
        'dias_derrame',
    ]


def test_validate_daily_dataset_raises_for_missing_columns():
    data = pd.DataFrame({'AH': ['AH-2012']})

    with pytest.raises(ValueError, match='Missing required columns'):
        validate_daily_dataset(data)


def test_compute_annual_feature_matrix_from_synthetic_data():
    config = load_pipeline_config()
    daily_data = build_synthetic_daily_data()

    features = compute_annual_feature_matrix(daily_data, config)

    assert features.shape == (2, 7)
    assert features['AH'].tolist() == ['AH-2012', 'AH-2013']

    first = features.loc[features['AH'] == 'AH-2012'].iloc[0]
    assert first['vol_min'] == pytest.approx(0.20)
    assert first['vol_amp'] == pytest.approx(0.30)
    assert first['inflow_avenida'] == pytest.approx(15.0)
    assert first['inflow_estiaje'] == pytest.approx(5.0)
    assert first['dias_derrame'] == pytest.approx(1)


def test_validate_feature_matrix_raises_for_missing_feature():
    config = load_pipeline_config()
    features = pd.DataFrame(
        {
            'AH': ['AH-2012'],
            'AH_YEAR': [2012],
            'vol_min': [0.1],
        }
    )

    with pytest.raises(ValueError, match='Missing required columns'):
        validate_feature_matrix(features, config)


def test_calculate_feature_correlations_returns_matrix_and_max_value():
    config = load_pipeline_config()
    daily_data = build_synthetic_daily_data()
    features = compute_annual_feature_matrix(daily_data, config)

    correlation, max_abs_correlation = calculate_feature_correlations(
        features,
        config,
    )

    assert correlation.shape == (5, 5)
    assert 0.0 <= max_abs_correlation <= 1.0


def test_generate_annual_features_returns_expected_years_if_file_exists():
    config = load_pipeline_config()
    raw_path = Path(config['rutas']['datos_crudos'])

    if not raw_path.exists():
        pytest.skip('Raw Excel file is not available in this environment.')

    features = generate_annual_features(config)

    assert features.shape == (14, 7)
    assert features['AH'].min() == 'AH-2012'
    assert features['AH'].max() == 'AH-2025'
    assert features[get_feature_columns(config)].notna().all().all()


def test_generate_annual_features_writes_csv_if_file_exists():
    config = load_pipeline_config()
    raw_path = Path(config['rutas']['datos_crudos'])

    if not raw_path.exists():
        pytest.skip('Raw Excel file is not available in this environment.')

    output_path = Path(config['rutas']['features'])
    features = generate_annual_features(config)

    assert output_path.exists()
    assert len(features) == 14
