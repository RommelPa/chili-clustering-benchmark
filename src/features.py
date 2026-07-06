from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config_loader import load_pipeline_config
from src.preprocessing import preprocess_daily_data


def get_feature_columns(config: dict[str, Any]) -> list[str]:
    """Return the configured annual feature column names."""
    return [feature['codigo'] for feature in config['features']['base']]


def validate_daily_dataset(data: pd.DataFrame) -> None:
    """Validate required columns in the processed daily dataset."""
    required_columns = [
        'FECHA',
        'AH_YEAR',
        'AH',
        'TEMPORADA',
        'INGCHINEGASA',
        'VOL_TOTAL_REL',
        'DERRAME_FLAG',
    ]

    missing = [column for column in required_columns if column not in data.columns]

    if missing:
        missing_text = ', '.join(missing)
        raise ValueError(f'Missing required columns in daily dataset: {missing_text}')

    required_seasons = {'avenida', 'estiaje'}
    observed_seasons = set(data['TEMPORADA'].dropna().unique())

    if not required_seasons.issubset(observed_seasons):
        raise ValueError(
            'Daily dataset must contain both avenida and estiaje seasons.'
        )


def compute_annual_feature_matrix(
    daily_data: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Compute annual hydrological-operational features from daily data."""
    validate_daily_dataset(daily_data)

    feature_columns = get_feature_columns(config)

    annual_storage = (
        daily_data.groupby(['AH', 'AH_YEAR'], as_index=False)
        .agg(
            vol_min=('VOL_TOTAL_REL', 'min'),
            vol_max=('VOL_TOTAL_REL', 'max'),
            dias_derrame=('DERRAME_FLAG', 'sum'),
        )
    )

    annual_storage['vol_amp'] = (
        annual_storage['vol_max'] - annual_storage['vol_min']
    )

    seasonal_inflow = (
        daily_data.pivot_table(
            index=['AH', 'AH_YEAR'],
            columns='TEMPORADA',
            values='INGCHINEGASA',
            aggfunc='mean',
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    missing_seasons = [
        season for season in ['avenida', 'estiaje']
        if season not in seasonal_inflow.columns
    ]

    if missing_seasons:
        missing_text = ', '.join(missing_seasons)
        raise ValueError(f'Missing seasonal inflow columns: {missing_text}')

    seasonal_inflow = seasonal_inflow.rename(
        columns={
            'avenida': 'inflow_avenida',
            'estiaje': 'inflow_estiaje',
        }
    )

    features = annual_storage.merge(
        seasonal_inflow[
            ['AH', 'AH_YEAR', 'inflow_avenida', 'inflow_estiaje']
        ],
        on=['AH', 'AH_YEAR'],
        how='left',
    )

    features = features[
        ['AH', 'AH_YEAR', *feature_columns]
    ].sort_values('AH_YEAR')

    features = features.reset_index(drop=True)

    validate_feature_matrix(features, config)

    return features


def validate_feature_matrix(
    features: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    """Validate the annual feature matrix before clustering."""
    feature_columns = get_feature_columns(config)
    required_columns = ['AH', 'AH_YEAR', *feature_columns]

    missing = [column for column in required_columns if column not in features.columns]

    if missing:
        missing_text = ', '.join(missing)
        raise ValueError(f'Missing required columns in feature matrix: {missing_text}')

    if features['AH'].duplicated().any():
        raise ValueError('Feature matrix contains duplicated AH labels.')

    if features[feature_columns].isna().any().any():
        raise ValueError('Feature matrix contains missing values.')

    non_numeric = [
        column for column in feature_columns
        if not pd.api.types.is_numeric_dtype(features[column])
    ]

    if non_numeric:
        non_numeric_text = ', '.join(non_numeric)
        raise ValueError(f'Feature columns must be numeric: {non_numeric_text}')


def calculate_feature_correlations(
    features: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, float]:
    """Calculate correlation matrix and maximum absolute pairwise correlation."""
    feature_columns = get_feature_columns(config)
    correlation = features[feature_columns].corr()

    upper_mask = np.triu(
        np.ones(correlation.shape, dtype=bool),
        k=1,
    )

    upper_values = correlation.abs().where(upper_mask)
    raw_max_correlation = upper_values.max().max()
    max_abs_correlation = float(np.clip(raw_max_correlation, 0.0, 1.0))

    return correlation, max_abs_correlation


def generate_annual_features(config: dict[str, Any] | str | Path) -> pd.DataFrame:
    """Run daily preprocessing and generate the annual feature matrix."""
    if isinstance(config, (str, Path)):
        config = load_pipeline_config(config)

    daily_data = preprocess_daily_data(config)
    features = compute_annual_feature_matrix(daily_data, config)

    output_path = Path(config['rutas']['features'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False, encoding='utf-8')

    correlation, max_abs_correlation = calculate_feature_correlations(
        features,
        config,
    )

    correlation_path = Path(config['rutas']['carpeta_tablas']) / (
        'feature_correlations.csv'
    )
    correlation_path.parent.mkdir(parents=True, exist_ok=True)
    correlation.to_csv(correlation_path, encoding='utf-8')

    print('[features] Matriz anual de features generada correctamente')
    print(f'[features] Años: {len(features)}')
    print(f'[features] Features: {len(get_feature_columns(config))}')
    print(f'[features] Máxima correlación absoluta: {max_abs_correlation:.3f}')

    return features


if __name__ == '__main__':
    pipeline_config = load_pipeline_config()
    feature_matrix = generate_annual_features(pipeline_config)

    print('[features] Rango AH:', feature_matrix['AH'].min(), '->', feature_matrix['AH'].max())
    print(feature_matrix)

