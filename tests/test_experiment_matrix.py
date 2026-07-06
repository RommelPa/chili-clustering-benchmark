import numpy as np
import pandas as pd
import pytest

from src.config_loader import load_experiments_config, load_pipeline_config
from src.experiment_matrix import (
    build_experiment_matrices,
    get_feature_sets,
    get_scalers,
    scale_feature_values,
    validate_feature_sets,
)


def build_synthetic_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'AH': ['AH-2012', 'AH-2013', 'AH-2014'],
            'AH_YEAR': [2012, 2013, 2014],
            'vol_min': [0.4, 0.3, 0.2],
            'vol_amp': [0.5, 0.4, 0.3],
            'inflow_avenida': [30.0, 20.0, 10.0],
            'inflow_estiaje': [12.0, 11.0, 10.0],
            'dias_derrame': [100, 50, 0],
        }
    )


def test_get_feature_sets_returns_base_operativa():
    config = load_experiments_config()

    feature_sets = get_feature_sets(config)

    assert 'base_operativa' in feature_sets
    assert feature_sets['base_operativa'] == [
        'vol_min',
        'vol_amp',
        'inflow_avenida',
        'inflow_estiaje',
        'dias_derrame',
    ]


def test_get_scalers_returns_zscore():
    config = load_experiments_config()

    scalers = get_scalers(config)

    assert scalers == ['zscore']


def test_validate_feature_sets_raises_for_missing_variable():
    config = load_experiments_config()
    features = build_synthetic_features().drop(columns=['vol_min'])

    with pytest.raises(ValueError, match='Missing variables'):
        validate_feature_sets(features, config)


def test_scale_feature_values_zscore_has_zero_mean_and_unit_std():
    features = build_synthetic_features()
    variables = ['vol_min', 'vol_amp', 'inflow_avenida']

    scaled = scale_feature_values(features, variables, 'zscore')

    assert scaled.shape == (3, 3)
    assert np.allclose(scaled.mean(axis=0), 0.0)
    assert np.allclose(scaled.std(axis=0), 1.0)


def test_scale_feature_values_raises_for_unknown_scaler():
    features = build_synthetic_features()

    with pytest.raises(ValueError, match='Unsupported scaler'):
        scale_feature_values(features, ['vol_min'], 'unknown')


def test_build_experiment_matrices_returns_configured_matrix():
    config = load_experiments_config()
    features = build_synthetic_features()

    matrices = build_experiment_matrices(features, config)

    assert len(matrices) == 1

    matrix = matrices[0]

    assert matrix.feature_set_name == 'base_operativa'
    assert matrix.scaler_name == 'zscore'
    assert matrix.values.shape == (3, 5)
    assert matrix.ah_labels == ['AH-2012', 'AH-2013', 'AH-2014']
    assert matrix.scaled_frame.shape == (3, 7)


def test_real_feature_columns_match_experiment_feature_set():
    pipeline_config = load_pipeline_config()
    experiments_config = load_experiments_config()

    configured_features = [
        feature['codigo']
        for feature in pipeline_config['features']['base']
    ]
    feature_sets = get_feature_sets(experiments_config)

    assert feature_sets['base_operativa'] == configured_features
