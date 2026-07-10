from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa

from src.config_loader import load_pipeline_config


def build_raw_expected_columns(config: dict[str, Any]) -> list[str]:
    """Build expected raw column names from pipeline configuration."""
    columns = ['FECHA', 'AÑO', 'MES', 'DIA']

    families = config['variables']['familias_por_embalse']
    suffixes = list(config['embalses']['sufijos'].keys())
    exceptions = config['variables'].get('excepciones', {})
    inverse_exceptions = {standard: raw for raw, standard in exceptions.items()}

    for family in families:
        prefix = family['prefijo']
        for suffix in suffixes:
            standard_name = f'{prefix}{suffix}'
            raw_name = inverse_exceptions.get(standard_name, standard_name)
            columns.append(raw_name)

    for special in config['variables'].get('especiales', []):
        columns.append(special['codigo'])

    return columns


def build_standard_expected_columns(config: dict[str, Any]) -> list[str]:
    """Build expected standardized column names after renaming exceptions."""
    raw_columns = build_raw_expected_columns(config)
    exceptions = config['variables'].get('excepciones', {})

    return [exceptions.get(column, column) for column in raw_columns]


def build_schema(config: dict[str, Any]) -> pa.DataFrameSchema:
    """Build a Pandera schema for the loaded raw dataset."""
    columns: dict[str, pa.Column] = {
        'FECHA': pa.Column(pa.DateTime, nullable=False, coerce=True),
        'AÑO': pa.Column(pa.Float, nullable=True, coerce=True),
        'MES': pa.Column(pa.Float, nullable=True, coerce=True),
        'DIA': pa.Column(pa.Float, nullable=True, coerce=True),
    }

    for column in build_standard_expected_columns(config):
        if column in columns:
            continue
        columns[column] = pa.Column(pa.Float, nullable=True, coerce=True)

    return pa.DataFrameSchema(columns=columns, strict=True, coerce=True)


def _reconstruct_header(raw_excel: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Reconstruct the dataset using the configured header and data rows."""
    header_row = int(config['archivo_fuente']['fila_encabezados'])
    data_start = int(config['archivo_fuente']['fila_inicio_datos'])

    headers = raw_excel.iloc[header_row].astype(str).str.strip().tolist()
    data = raw_excel.iloc[data_start:].copy()
    data.columns = headers

    return data.reset_index(drop=True)


def _select_expected_columns(
    data: pd.DataFrame,
    expected_columns: list[str],
) -> pd.DataFrame:
    """Select expected columns and raise an error if any required column is missing."""
    missing = [column for column in expected_columns if column not in data.columns]

    if missing:
        missing_text = ', '.join(missing)
        raise ValueError(f'Missing expected columns in raw file: {missing_text}')

    return data.loc[:, expected_columns].copy()


def _clean_types(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert FECHA and numeric columns, returning a small cleaning log."""
    initial_rows = len(data)

    data['FECHA'] = pd.to_datetime(data['FECHA'], errors='coerce')
    invalid_dates = int(data['FECHA'].isna().sum())
    data = data.loc[data['FECHA'].notna()].copy().reset_index(drop=True)

    lost_numeric_cells = 0
    numeric_columns = [column for column in data.columns if column != 'FECHA']

    for column in numeric_columns:
        original_not_null = data[column].notna()
        converted = pd.to_numeric(data[column], errors='coerce')
        lost_numeric_cells += int((original_not_null & converted.isna()).sum())
        data[column] = converted

    log = {
        'initial_rows': initial_rows,
        'invalid_date_rows_removed': invalid_dates,
        'final_rows': len(data),
        'lost_numeric_cells': lost_numeric_cells,
    }

    return data, log


def _write_cleaning_log(log: dict[str, Any], config: dict[str, Any]) -> None:
    """Write a lightweight cleaning log to disk."""
    path = Path(config['rutas']['log_limpieza'])
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        'Log de carga de datos crudos',
        '============================',
        '',
        f'Filas iniciales: {log["initial_rows"]}',
        f'Filas descartadas por FECHA inválida: {log["invalid_date_rows_removed"]}',
        f'Filas finales: {log["final_rows"]}',
        f'Celdas numéricas perdidas por conversión: {log["lost_numeric_cells"]}',
        '',
    ]

    path.write_text('\n'.join(lines), encoding='utf-8')


def load_raw_data(config: dict[str, Any] | str | Path) -> pd.DataFrame:
    """Load, clean and validate the raw BDREPRESAS Excel file."""
    if isinstance(config, (str, Path)):
        config = load_pipeline_config(config)

    path = Path(config['rutas']['datos_crudos'])
    if not path.exists():
        raise FileNotFoundError(f'Raw data file not found: {path}')

    source = config['archivo_fuente']
    raw_excel = pd.read_excel(
        path,
        sheet_name=source['hoja_datos'],
        header=None,
        engine='openpyxl',
    )

    expected_columns = build_raw_expected_columns(config)
    data = _reconstruct_header(raw_excel, config)
    data = _select_expected_columns(data, expected_columns)
    data, cleaning_log = _clean_types(data)

    exceptions = config['variables'].get('excepciones', {})
    data = data.rename(columns=exceptions)

    schema = build_schema(config)
    validated = schema.validate(data)

    _write_cleaning_log(cleaning_log, config)

    return validated


if __name__ == '__main__':
    pipeline_config = load_pipeline_config()
    df = load_raw_data(pipeline_config)

    print('[data_loader] Archivo crudo cargado correctamente')
    print(f'[data_loader] Filas: {len(df):,}')
    print(f'[data_loader] Columnas: {df.shape[1]}')
    print(
        '[data_loader] Rango de fechas: '
        f'{df["FECHA"].min().date()} -> {df["FECHA"].max().date()}'
    )
