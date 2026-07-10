from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOG_METRIC_MAP = {
    'Filas iniciales': 'initial_rows',
    'Filas descartadas por FECHA inválida': 'invalid_date_rows_removed',
    'Filas finales': 'raw_final_rows',
    'Celdas numéricas perdidas por conversión': 'lost_numeric_cells',
}


def get_configured_feature_columns(config: dict[str, Any]) -> list[str]:
    """Return feature column names configured for clustering."""
    return [feature['codigo'] for feature in config['features']['base']]


def parse_cleaning_log(log_path: Path) -> dict[str, str]:
    """Parse the lightweight raw data cleaning log."""
    if not log_path.exists():
        return {}

    parsed: dict[str, str] = {}
    content = log_path.read_text(encoding='utf-8').replace('\\n', '\n')

    for line in content.splitlines():
        if ':' not in line:
            continue

        key, value = line.split(':', maxsplit=1)
        metric = LOG_METRIC_MAP.get(key.strip())

        if metric is not None:
            parsed[metric] = value.strip()

    return parsed


def calculate_max_abs_correlation(correlation: pd.DataFrame) -> float:
    """Calculate maximum absolute pairwise correlation from a matrix."""
    numeric_correlation = correlation.apply(pd.to_numeric, errors='coerce')
    upper_mask = np.triu(
        np.ones(numeric_correlation.shape, dtype=bool),
        k=1,
    )
    upper_values = numeric_correlation.abs().where(upper_mask)
    max_value = upper_values.max().max()

    return float(max_value)


def build_summary_row(
    section: str,
    metric: str,
    value: Any,
    unit: str = '',
) -> dict[str, str]:
    """Build a normalized summary row."""
    return {
        'section': section,
        'metric': metric,
        'value': str(value),
        'unit': unit,
    }


def build_data_processing_summary(
    daily_data: pd.DataFrame,
    features: pd.DataFrame,
    correlations: pd.DataFrame,
    config: dict[str, Any],
    cleaning_log: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build a reproducible processing summary table."""
    cleaning_log = cleaning_log or {}
    feature_columns = get_configured_feature_columns(config)

    daily_dates = pd.to_datetime(daily_data['FECHA'])
    hydrological_years = features['AH_YEAR'].astype(int)

    rows: list[dict[str, str]] = []

    for metric in [
        'initial_rows',
        'invalid_date_rows_removed',
        'raw_final_rows',
        'lost_numeric_cells',
    ]:
        if metric in cleaning_log:
            rows.append(
                build_summary_row(
                    section='raw_loading',
                    metric=metric,
                    value=cleaning_log[metric],
                )
            )

    rows.extend(
        [
            build_summary_row(
                'daily_preprocessing',
                'daily_rows',
                len(daily_data),
                'rows',
            ),
            build_summary_row(
                'daily_preprocessing',
                'daily_columns',
                daily_data.shape[1],
                'columns',
            ),
            build_summary_row(
                'daily_preprocessing',
                'start_date',
                daily_dates.min().date(),
            ),
            build_summary_row(
                'daily_preprocessing',
                'end_date',
                daily_dates.max().date(),
            ),
            build_summary_row(
                'daily_preprocessing',
                'hydrological_year_start',
                f'AH-{hydrological_years.min()}',
            ),
            build_summary_row(
                'daily_preprocessing',
                'hydrological_year_end',
                f'AH-{hydrological_years.max()}',
            ),
            build_summary_row(
                'daily_preprocessing',
                'hydrological_year_count',
                features['AH'].nunique(),
                'years',
            ),
            build_summary_row(
                'annual_features',
                'feature_rows',
                len(features),
                'rows',
            ),
            build_summary_row(
                'annual_features',
                'feature_count',
                len(feature_columns),
                'features',
            ),
            build_summary_row(
                'annual_features',
                'feature_names',
                ', '.join(feature_columns),
            ),
            build_summary_row(
                'annual_features',
                'max_abs_pairwise_correlation',
                f'{calculate_max_abs_correlation(correlations):.3f}',
            ),
        ]
    )

    return pd.DataFrame(rows)


def get_data_summary_output_path(config: dict[str, Any]) -> Path:
    """Return output path for the data processing summary."""
    return Path(config['rutas']['carpeta_tablas']) / (
        'data_processing_summary.csv'
    )


def load_required_outputs(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load processed daily data, annual features and feature correlations."""
    daily_data = pd.read_csv(config['rutas']['dataset_procesado'])
    features = pd.read_csv(config['rutas']['features'])
    correlations = pd.read_csv(
        Path(config['rutas']['carpeta_tablas'])
        / 'feature_correlations.csv',
        index_col=0,
    )

    return daily_data, features, correlations


def save_data_processing_summary(
    summary: pd.DataFrame,
    config: dict[str, Any],
) -> Path:
    """Save the data processing summary table."""
    output_path = get_data_summary_output_path(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False, encoding='utf-8')

    return output_path


def run_data_summary(config: dict[str, Any]) -> pd.DataFrame:
    """Generate and save the data processing summary table."""
    daily_data, features, correlations = load_required_outputs(config)
    cleaning_log = parse_cleaning_log(Path(config['rutas']['log_limpieza']))

    summary = build_data_processing_summary(
        daily_data=daily_data,
        features=features,
        correlations=correlations,
        config=config,
        cleaning_log=cleaning_log,
    )

    output_path = save_data_processing_summary(summary, config)

    print('[data_summary] Resumen de procesamiento generado')
    print(f'[data_summary] Tabla: {output_path}')
    print(summary.to_string(index=False))

    return summary


if __name__ == '__main__':
    from src.config_loader import load_pipeline_config

    pipeline_config = load_pipeline_config()
    run_data_summary(pipeline_config)
