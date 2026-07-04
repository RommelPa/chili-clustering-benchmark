from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config_loader import load_pipeline_config
from src.data_loader import load_raw_data


def filter_study_period(data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Filter raw data to the configured study period."""
    start = pd.Timestamp(config['dataset']['periodo_inicio'])
    end = pd.Timestamp(config['dataset']['periodo_fin'])

    filtered = data.loc[
        (data['FECHA'] >= start) & (data['FECHA'] <= end)
    ].copy()

    return filtered.reset_index(drop=True)


def assign_hydrological_year(
    data: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Assign hydrological-operational year labels."""
    result = data.copy()
    start_month = int(config['dataset']['ano_hidrologico']['mes_inicio'])

    calendar_year = result['FECHA'].dt.year
    calendar_month = result['FECHA'].dt.month

    ah_year = calendar_year.where(calendar_month < start_month, calendar_year + 1)

    result['AH_YEAR'] = ah_year.astype(int)
    result['AH'] = 'AH-' + result['AH_YEAR'].astype(str)

    return result


def assign_operational_season(
    data: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Assign avenida or estiaje season according to configured months."""
    result = data.copy()
    avenida_months = set(config['dataset']['ventanas']['avenida']['meses'])

    result['TEMPORADA'] = result['FECHA'].dt.month.map(
        lambda month: 'avenida' if month in avenida_months else 'estiaje'
    )

    return result


def get_columns_by_prefix(config: dict[str, Any], prefix: str) -> list[str]:
    """Return standardized reservoir columns for a given variable prefix."""
    suffixes = list(config['embalses']['sufijos'].keys())
    return [f'{prefix}{suffix}' for suffix in suffixes]


def calculate_daily_operational_aggregates(
    data: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Calculate daily system-level operational aggregates."""
    result = data.copy()

    volume_columns = get_columns_by_prefix(config, 'VOLUTIL')
    discharge_columns = get_columns_by_prefix(config, 'DESCARGA')
    spill_columns = get_columns_by_prefix(config, 'REBOSE')

    total_capacity_hm3 = sum(
        float(value)
        for value in config['embalses']['capacidad_util_operativa_hm3'].values()
    )

    result['VOL_TOTAL_HM3'] = result[volume_columns].sum(axis=1) / 1_000_000
    result['VOL_TOTAL_REL'] = result['VOL_TOTAL_HM3'] / total_capacity_hm3
    result['DESC_TOTAL_M3S'] = result[discharge_columns].sum(axis=1)
    result['REBOSE_TOTAL_M3S'] = result[spill_columns].sum(axis=1)
    result['DERRAME_FLAG'] = result['REBOSE_TOTAL_M3S'].gt(0).astype(int)

    return result


def select_processed_columns(
    data: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Select the columns that will be kept in the processed daily dataset."""
    volume_columns = get_columns_by_prefix(config, 'VOLUTIL')
    discharge_columns = get_columns_by_prefix(config, 'DESCARGA')
    spill_columns = get_columns_by_prefix(config, 'REBOSE')

    base_columns = [
        'FECHA',
        'AH_YEAR',
        'AH',
        'TEMPORADA',
        'INGCHINEGASA',
        'VOL_TOTAL_HM3',
        'VOL_TOTAL_REL',
        'DESC_TOTAL_M3S',
        'REBOSE_TOTAL_M3S',
        'DERRAME_FLAG',
    ]

    selected_columns = (
        base_columns
        + volume_columns
        + discharge_columns
        + spill_columns
    )

    return data.loc[:, selected_columns].copy()


def preprocess_daily_data(config: dict[str, Any] | str | Path) -> pd.DataFrame:
    """Run the full daily preprocessing step."""
    if isinstance(config, (str, Path)):
        config = load_pipeline_config(config)

    raw_data = load_raw_data(config)

    processed = filter_study_period(raw_data, config)
    processed = assign_hydrological_year(processed, config)
    processed = assign_operational_season(processed, config)
    processed = calculate_daily_operational_aggregates(processed, config)
    processed = select_processed_columns(processed, config)

    output_path = Path(config['rutas']['dataset_procesado'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False, encoding='utf-8')

    return processed


if __name__ == '__main__':
    pipeline_config = load_pipeline_config()
    df = preprocess_daily_data(pipeline_config)

    print('[preprocessing] Dataset diario procesado correctamente')
    print(f'[preprocessing] Filas: {len(df):,}')
    print(f'[preprocessing] Columnas: {df.shape[1]}')
    print(
        '[preprocessing] Rango de fechas: '
        f'{df["FECHA"].min().date()} -> {df["FECHA"].max().date()}'
    )
    print(
        '[preprocessing] Años hidrológico-operativos: '
        f'{df["AH"].min()} -> {df["AH"].max()}'
    )
