from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def get_table_output_dir(experiments_config: dict[str, Any]) -> Path:
    """Return table output directory."""
    benchmark_path = Path(
        experiments_config['salidas']['tablas']['benchmark_runs']
    )

    return benchmark_path.parent


def get_report_output_dir(experiments_config: dict[str, Any]) -> Path:
    """Return report output directory with safe fallback."""
    report_config = experiments_config.get('salidas', {}).get('reportes')

    if isinstance(report_config, dict):
        directory = report_config.get('directorio') or report_config.get('dir')
        if directory is not None:
            return Path(directory)

    if isinstance(report_config, str):
        return Path(report_config)

    return Path('outputs/reports')


def get_report_output_path(experiments_config: dict[str, Any]) -> Path:
    """Return methodology and results report path."""
    return get_report_output_dir(experiments_config) / (
        'methodology_results_summary.md'
    )


def to_bool_series(values: pd.Series) -> pd.Series:
    """Convert mixed boolean-like values to boolean."""
    return values.astype(str).str.lower().isin(['true', '1', 'yes', 'y'])


def summarize_benchmark_outputs(
    benchmark: pd.DataFrame,
    final_runs: pd.DataFrame,
    final_partitions: pd.DataFrame,
) -> dict[str, Any]:
    """Build a compact benchmark summary."""
    summary: dict[str, Any] = {
        'n_total_configurations': len(benchmark),
        'n_ranked_runs': len(final_runs),
        'n_unique_partitions': len(final_partitions),
    }

    if 'status' in benchmark.columns:
        summary['n_successful_configurations'] = int(
            (benchmark['status'].astype(str) == 'success').sum()
        )
    else:
        summary['n_successful_configurations'] = len(benchmark)

    if 'is_valid_partition' in benchmark.columns:
        summary['n_valid_partitions'] = int(
            to_bool_series(benchmark['is_valid_partition']).sum()
        )
    elif 'valid_partition' in benchmark.columns:
        summary['n_valid_partitions'] = int(
            to_bool_series(benchmark['valid_partition']).sum()
        )
    else:
        summary['n_valid_partitions'] = len(final_runs)

    return summary


def format_value(value: Any, digits: int = 3) -> str:
    """Format values for Markdown tables."""
    if pd.isna(value):
        return ''

    if isinstance(value, float):
        return f'{value:.{digits}f}'

    return str(value)


def dataframe_to_markdown(
    data: pd.DataFrame,
    columns: list[str] | None = None,
    max_rows: int | None = None,
    digits: int = 3,
) -> str:
    """Render a DataFrame as a simple Markdown table."""
    if columns is None:
        columns = data.columns.tolist()

    selected = data[columns].copy()

    if max_rows is not None:
        selected = selected.head(max_rows)

    header = '| ' + ' | '.join(columns) + ' |'
    separator = '| ' + ' | '.join(['---'] * len(columns)) + ' |'

    rows = []
    for _, row in selected.iterrows():
        values = [
            format_value(row[column], digits=digits)
            for column in columns
        ]
        rows.append('| ' + ' | '.join(values) + ' |')

    return '\n'.join([header, separator, *rows])


def select_top_partition(final_partitions: pd.DataFrame) -> pd.Series:
    """Select the best final partition."""
    if final_partitions.empty:
        raise ValueError('Final partition ranking is empty.')

    if 'final_partition_rank' in final_partitions.columns:
        return final_partitions.sort_values('final_partition_rank').iloc[0]

    return final_partitions.sort_values(
        'final_score',
        ascending=False,
    ).iloc[0]


def build_years_by_cluster(membership: pd.DataFrame) -> pd.DataFrame:
    """Summarize years assigned to each cluster."""
    return (
        membership.groupby('cluster_id')
        .agg(
            n_years=('AH', 'count'),
            years=('AH', lambda values: ', '.join(values.astype(str))),
        )
        .reset_index()
        .sort_values('cluster_id')
    )


def get_profile_columns(profiles: pd.DataFrame) -> list[str]:
    """Return profile columns for the report."""
    mean_columns = [
        column for column in profiles.columns
        if column.endswith('_mean')
    ]

    return ['cluster_id', 'n_years', *mean_columns]


def build_methodology_section(summary: dict[str, Any]) -> str:
    """Build methodology section in Spanish."""
    return f"""
## Metodología reproducible

Se construyó un flujo experimental reproducible para evaluar tipologías
hidrológico-operativas anuales mediante algoritmos de clustering no
supervisado. La matriz de análisis fue generada a partir de registros diarios
agregados por año hidrológico-operativo.

El diseño experimental comparó múltiples familias de algoritmos y combinaciones
de hiperparámetros. Las corridas fueron evaluadas mediante métricas internas,
restricciones de partición, equivalencia entre particiones y estabilidad por
submuestreo.

Resumen del flujo experimental:

- Configuraciones evaluadas: {summary['n_total_configurations']}
- Configuraciones ejecutadas correctamente: {summary['n_successful_configurations']}
- Particiones válidas para ranking: {summary['n_valid_partitions']}
- Corridas incluidas en el ranking final: {summary['n_ranked_runs']}
- Particiones únicas identificadas: {summary['n_unique_partitions']}

La selección final no se basó únicamente en un algoritmo individual. Se integró
la calidad interna de la partición, la estabilidad por submuestreo y el soporte
de configuraciones equivalentes.
""".strip()


def build_results_section(
    final_partitions: pd.DataFrame,
    membership: pd.DataFrame,
    profiles: pd.DataFrame,
    differences: pd.DataFrame,
) -> str:
    """Build preliminary results section in Spanish."""
    top_partition = select_top_partition(final_partitions)
    years_by_cluster = build_years_by_cluster(membership)

    profile_columns = get_profile_columns(profiles)

    partition_table = dataframe_to_markdown(
        final_partitions,
        columns=[
            'final_partition_rank',
            'partition_id',
            'representative_run_id',
            'representative_algorithm',
            'final_score',
            'ranking_score',
            'stability_score',
            'n_clusters',
            'n_equivalent_runs',
            'n_supporting_algorithms',
        ],
        max_rows=10,
    )

    years_table = dataframe_to_markdown(
        years_by_cluster,
        columns=['cluster_id', 'n_years', 'years'],
    )

    profiles_table = dataframe_to_markdown(
        profiles,
        columns=profile_columns,
    )

    differences_table = dataframe_to_markdown(
        differences,
        columns=[
            'cluster_id',
            'feature',
            'cluster_mean',
            'global_mean',
            'difference',
            'standardized_difference',
            'direction',
        ],
        max_rows=10,
    )

    return f"""
## Resultados preliminares

La partición final mejor posicionada fue `{top_partition['partition_id']}`,
representada por la corrida `{top_partition['representative_run_id']}` del
algoritmo `{top_partition['representative_algorithm']}`.

Esta partición obtuvo un score final integrado de
{format_value(top_partition['final_score'], digits=4)}, con score de ranking
interno de {format_value(top_partition['ranking_score'], digits=4)} y score de
estabilidad de {format_value(top_partition['stability_score'], digits=4)}.

La partición seleccionada contiene {top_partition['n_clusters']} clústeres,
aparece en {top_partition['n_equivalent_runs']} configuraciones equivalentes y
está soportada por {top_partition['n_supporting_algorithms']} familias o tipos
de algoritmos.

### Ranking final de particiones

{partition_table}

### Años asignados a cada clúster

{years_table}

### Perfiles promedio por clúster

{profiles_table}

### Diferencias más relevantes frente a la media global

{differences_table}
""".strip()


def build_interpretation_section() -> str:
    """Build interpretation section in Spanish."""
    return """
## Interpretación operativa preliminar

La partición seleccionada separa los años hidrológico-operativos en dos grupos
principales. El primer grupo presenta valores promedio superiores en las
variables de almacenamiento relativo, aportes medios y días con derrame. El
segundo grupo presenta valores inferiores respecto a la media global,
especialmente en caudales de estiaje, volumen mínimo relativo y días con
derrame.

En términos operativos, esta estructura puede interpretarse preliminarmente como
una separación entre años de mayor disponibilidad hídrica-operativa y años de
condición más restrictiva.

Esta interpretación debe tratarse como preliminar. La validación final requiere
revisión experta del dominio, contraste con eventos hidrológicos conocidos y
discusión de las implicancias operativas para el sistema regulado Chili.
""".strip()


def build_methodology_results_report(
    benchmark: pd.DataFrame,
    final_runs: pd.DataFrame,
    final_partitions: pd.DataFrame,
    membership: pd.DataFrame,
    profiles: pd.DataFrame,
    differences: pd.DataFrame,
) -> str:
    """Build complete Markdown report."""
    summary = summarize_benchmark_outputs(
        benchmark=benchmark,
        final_runs=final_runs,
        final_partitions=final_partitions,
    )

    sections = [
        '# Resumen metodológico y resultados preliminares',
        build_methodology_section(summary),
        build_results_section(
            final_partitions=final_partitions,
            membership=membership,
            profiles=profiles,
            differences=differences,
        ),
        build_interpretation_section(),
    ]

    return '\n\n'.join(sections) + '\n'


def load_required_outputs(
    experiments_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load required outputs for the report."""
    table_dir = get_table_output_dir(experiments_config)

    benchmark = pd.read_csv(table_dir / 'benchmark_runs.csv')
    final_runs = pd.read_csv(table_dir / 'final_run_ranking.csv')
    final_partitions = pd.read_csv(table_dir / 'final_partition_ranking.csv')
    membership = pd.read_csv(table_dir / 'selected_partition_membership.csv')
    profiles = pd.read_csv(table_dir / 'selected_cluster_profiles.csv')
    differences = pd.read_csv(
        table_dir / 'selected_cluster_feature_differences.csv'
    )

    return (
        benchmark,
        final_runs,
        final_partitions,
        membership,
        profiles,
        differences,
    )


def save_report(content: str, output_path: Path) -> Path:
    """Save report content to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding='utf-8')

    return output_path


def generate_report(experiments_config: dict[str, Any]) -> Path:
    """Generate methodology and preliminary results report."""
    (
        benchmark,
        final_runs,
        final_partitions,
        membership,
        profiles,
        differences,
    ) = load_required_outputs(experiments_config)

    report = build_methodology_results_report(
        benchmark=benchmark,
        final_runs=final_runs,
        final_partitions=final_partitions,
        membership=membership,
        profiles=profiles,
        differences=differences,
    )

    output_path = save_report(
        content=report,
        output_path=get_report_output_path(experiments_config),
    )

    print('[reporting] Reporte metodológico generado correctamente')
    print(f'[reporting] Ruta: {output_path}')

    return output_path


if __name__ == '__main__':
    from src.config_loader import load_experiments_config

    experiments_config = load_experiments_config()
    generate_report(experiments_config)
