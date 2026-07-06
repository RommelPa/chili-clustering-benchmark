from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd


def get_table_output_dir(experiments_config: dict[str, Any]) -> Path:
    """Return table output directory."""
    benchmark_path = Path(
        experiments_config['salidas']['tablas']['benchmark_runs']
    )

    return benchmark_path.parent


def get_figure_output_dir(experiments_config: dict[str, Any]) -> Path:
    """Return figure output directory with safe fallback."""
    figure_config = experiments_config.get('salidas', {}).get('figuras')

    if isinstance(figure_config, dict):
        directory = figure_config.get('directorio') or figure_config.get('dir')
        if directory is not None:
            return Path(directory)

    if isinstance(figure_config, str):
        return Path(figure_config)

    return Path('outputs/figures')


def load_required_outputs(
    experiments_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load final ranking and interpretation outputs."""
    table_dir = get_table_output_dir(experiments_config)

    final_partitions = pd.read_csv(table_dir / 'final_partition_ranking.csv')
    differences = pd.read_csv(
        table_dir / 'selected_cluster_feature_differences.csv'
    )
    membership = pd.read_csv(table_dir / 'selected_partition_membership.csv')

    return final_partitions, differences, membership


def prepare_partition_score_data(
    final_partitions: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Prepare top partition score data for plotting."""
    required_columns = [
        'final_partition_rank',
        'partition_id',
        'final_score',
        'stability_score',
        'ranking_score',
        'n_clusters',
    ]

    missing = [
        column for column in required_columns
        if column not in final_partitions.columns
    ]

    if missing:
        raise ValueError(f'Missing partition ranking columns: {missing}')

    return (
        final_partitions[required_columns]
        .sort_values('final_partition_rank')
        .head(top_n)
        .reset_index(drop=True)
    )


def prepare_cluster_difference_matrix(
    differences: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare standardized feature differences by cluster."""
    required_columns = [
        'cluster_id',
        'feature',
        'standardized_difference',
    ]

    missing = [
        column for column in required_columns
        if column not in differences.columns
    ]

    if missing:
        raise ValueError(f'Missing difference columns: {missing}')

    matrix = differences.pivot(
        index='feature',
        columns='cluster_id',
        values='standardized_difference',
    ).fillna(0.0)

    feature_order = (
        matrix.abs()
        .max(axis=1)
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    return matrix.loc[feature_order]


def prepare_year_typology_data(membership: pd.DataFrame) -> pd.DataFrame:
    """Prepare annual cluster membership for plotting."""
    required_columns = ['AH', 'AH_YEAR', 'cluster_id']

    missing = [
        column for column in required_columns
        if column not in membership.columns
    ]

    if missing:
        raise ValueError(f'Missing membership columns: {missing}')

    data = membership[required_columns].copy()
    data = data.sort_values('AH_YEAR').reset_index(drop=True)

    cluster_order = sorted(data['cluster_id'].astype(str).unique())
    cluster_to_position = {
        cluster_id: index
        for index, cluster_id in enumerate(cluster_order)
    }

    data['cluster_position'] = data['cluster_id'].map(cluster_to_position)

    return data


def plot_final_partition_scores(
    final_partitions: pd.DataFrame,
    output_path: Path,
    top_n: int = 10,
) -> Path:
    """Plot final scores for top-ranked partitions."""
    data = prepare_partition_score_data(final_partitions, top_n=top_n)
    labels = (
        data['partition_id'].astype(str)
        + '\nK='
        + data['n_clusters'].astype(str)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, data['final_score'])
    ax.set_ylim(0, 1)
    ax.set_xlabel('Partición')
    ax.set_ylabel('Score final integrado')
    ax.set_title('Ranking final de particiones')
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def plot_cluster_difference_profile(
    differences: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plot standardized cluster differences against the global mean."""
    matrix = prepare_cluster_difference_matrix(differences)

    fig, ax = plt.subplots(figsize=(10, 5))
    matrix.plot(kind='bar', ax=ax)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel('Variable')
    ax.set_ylabel('Diferencia estandarizada frente a la media global')
    ax.set_title('Perfil diferencial de los clústeres seleccionados')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Clúster')
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def plot_year_typology(
    membership: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plot annual hydrological-operational typology."""
    data = prepare_year_typology_data(membership)
    cluster_order = sorted(data['cluster_id'].astype(str).unique())

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(data['AH_YEAR'], data['cluster_position'])

    for _, row in data.iterrows():
        label = str(row['AH']).replace('AH-', '')
        ax.annotate(
            label,
            (row['AH_YEAR'], row['cluster_position']),
            textcoords='offset points',
            xytext=(0, 6),
            ha='center',
            fontsize=8,
        )

    ax.set_yticks(range(len(cluster_order)))
    ax.set_yticklabels(cluster_order)
    ax.set_xlabel('Año hidrológico-operativo')
    ax.set_ylabel('Tipología')
    ax.set_title('Asignación anual a la partición seleccionada')
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def get_figure_paths(
    experiments_config: dict[str, Any],
) -> dict[str, Path]:
    """Return output paths for result figures."""
    figure_dir = get_figure_output_dir(experiments_config)

    return {
        'final_partition_scores': figure_dir / 'final_partition_scores.png',
        'cluster_difference_profile': (
            figure_dir / 'cluster_difference_profile.png'
        ),
        'year_typology': figure_dir / 'year_typology.png',
    }


def generate_figures(
    experiments_config: dict[str, Any],
) -> dict[str, Path]:
    """Generate all result figures."""
    final_partitions, differences, membership = load_required_outputs(
        experiments_config
    )
    figure_paths = get_figure_paths(experiments_config)

    plot_final_partition_scores(
        final_partitions=final_partitions,
        output_path=figure_paths['final_partition_scores'],
    )
    plot_cluster_difference_profile(
        differences=differences,
        output_path=figure_paths['cluster_difference_profile'],
    )
    plot_year_typology(
        membership=membership,
        output_path=figure_paths['year_typology'],
    )

    print('[figures] Figuras generadas correctamente')
    for name, path in figure_paths.items():
        print(f'[figures] {name}: {path}')

    return figure_paths


if __name__ == '__main__':
    from src.config_loader import load_experiments_config

    experiments_config = load_experiments_config()
    generate_figures(experiments_config)
