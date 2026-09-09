from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA


ALGORITHM_LABELS = {
    'spectral': 'Spectral',
    'birch': 'BIRCH',
    'gaussian_mixture': 'Gaussian mixture',
    'agglomerative': 'Agglomerative',
    'kmeans': 'K-means',
    'dbscan': 'DBSCAN',
}


def format_algorithm_label(algorithm_name: str) -> str:
    """Format algorithm name for figures."""
    return ALGORITHM_LABELS.get(str(algorithm_name), str(algorithm_name))


def format_cluster_label(cluster_id: str) -> str:
    """Format cluster label for figures."""
    return str(cluster_id).replace('_', ' ').title()


def format_grouping_label(grouping_id: str) -> str:
    """Format internal partition id as public grouping label."""
    numeric_id = str(grouping_id).split('_')[-1]

    if numeric_id.isdigit():
        return f'A-{int(numeric_id):03d}'

    return str(grouping_id)


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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load final ranking, interpretation, and algorithm outputs."""
    table_dir = get_table_output_dir(experiments_config)

    final_partitions = pd.read_csv(table_dir / 'final_partition_ranking.csv')
    differences = pd.read_csv(
        table_dir / 'selected_cluster_feature_differences.csv'
    )
    membership = pd.read_csv(table_dir / 'selected_partition_membership.csv')
    algorithm_summary = pd.read_csv(
        table_dir / 'algorithm_performance_summary.csv'
    )

    return final_partitions, differences, membership, algorithm_summary


def prepare_partition_score_data(
    final_partitions: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Prepare top grouping score data for plotting."""
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
        raise ValueError(f'Missing final ranking columns: {missing}')

    data = (
        final_partitions[required_columns]
        .sort_values('final_partition_rank')
        .head(top_n)
        .reset_index(drop=True)
    )

    data['grouping_label'] = data['partition_id'].map(format_grouping_label)

    return data


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


def prepare_algorithm_score_data(
    algorithm_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare best algorithm scores for plotting."""
    required_columns = [
        'algorithm_name',
        'best_final_score',
        'best_stability_score',
        'best_n_clusters',
        'best_partition_id',
    ]

    missing = [
        column for column in required_columns
        if column not in algorithm_summary.columns
    ]

    if missing:
        raise ValueError(f'Missing algorithm score columns: {missing}')

    data = (
        algorithm_summary[required_columns]
        .dropna(subset=['best_final_score'])
        .sort_values('best_final_score', ascending=False)
        .reset_index(drop=True)
    )

    data['algorithm_label'] = data['algorithm_name'].map(format_algorithm_label)

    return data


def prepare_valid_partition_data(
    algorithm_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare retained grouping counts by algorithm."""
    required_columns = [
        'algorithm_name',
        'n_configurations',
        'n_valid_partitions',
        'valid_partition_rate',
    ]

    missing = [
        column for column in required_columns
        if column not in algorithm_summary.columns
    ]

    if missing:
        raise ValueError(f'Missing retained grouping columns: {missing}')

    data = (
        algorithm_summary[required_columns]
        .sort_values(
            by=['n_valid_partitions', 'n_configurations'],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )

    data['algorithm_label'] = data['algorithm_name'].map(format_algorithm_label)

    return data


def load_feature_correlation_data(
    experiments_config: dict[str, Any],
) -> pd.DataFrame:
    """Load feature correlation matrix."""
    table_dir = get_table_output_dir(experiments_config)
    correlation_path = table_dir / 'feature_correlations.csv'

    return pd.read_csv(correlation_path, index_col=0)


def load_standardized_feature_matrix(
    experiments_config: dict[str, Any],
) -> pd.DataFrame:
    """Load standardized annual feature matrix."""
    table_dir = get_table_output_dir(experiments_config)
    matrix_path = table_dir / 'matrix_base_operativa_zscore.csv'

    return pd.read_csv(matrix_path)


def prepare_feature_correlation_matrix(
    correlation_df: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare feature correlation matrix for plotting."""
    if correlation_df.empty:
        raise ValueError('Feature correlation matrix is empty.')

    if correlation_df.shape[0] != correlation_df.shape[1]:
        raise ValueError('Feature correlation matrix must be square.')

    feature_labels = {
        'vol_min': 'Min. storage',
        'vol_amp': 'Storage amplitude',
        'inflow_avenida': 'Wet-season inflow',
        'inflow_estiaje': 'Dry-season inflow',
        'dias_derrame': 'Spill days',
    }

    matrix = correlation_df.apply(pd.to_numeric, errors='coerce')

    if matrix.isna().any().any():
        raise ValueError('Feature correlation matrix must be numeric.')

    matrix.index = [
        feature_labels.get(feature, feature)
        for feature in matrix.index
    ]
    matrix.columns = [
        feature_labels.get(feature, feature)
        for feature in matrix.columns
    ]

    return matrix


def prepare_selected_grouping_pca_data(
    feature_matrix: pd.DataFrame,
    membership: pd.DataFrame,
) -> tuple[pd.DataFrame, list[float]]:
    """Project selected grouping into two PCA components."""
    required_matrix_columns = ['AH', 'AH_YEAR']
    required_membership_columns = ['AH', 'cluster_id']

    missing_matrix = [
        column for column in required_matrix_columns
        if column not in feature_matrix.columns
    ]
    missing_membership = [
        column for column in required_membership_columns
        if column not in membership.columns
    ]

    if missing_matrix:
        raise ValueError(f'Missing feature matrix columns: {missing_matrix}')

    if missing_membership:
        raise ValueError(f'Missing membership columns: {missing_membership}')

    feature_columns = [
        column for column in feature_matrix.columns
        if column not in {'AH', 'AH_YEAR'}
    ]

    if len(feature_columns) < 2:
        raise ValueError('At least two feature columns are required for PCA.')

    values = feature_matrix[feature_columns].apply(
        pd.to_numeric,
        errors='coerce',
    )

    if values.isna().any().any():
        raise ValueError('Feature matrix contains non-numeric values.')

    pca = PCA(n_components=2)
    components = pca.fit_transform(values)

    pca_data = feature_matrix[['AH', 'AH_YEAR']].copy()
    pca_data['PC1'] = components[:, 0]
    pca_data['PC2'] = components[:, 1]

    cluster_data = membership[['AH', 'cluster_id']].copy()

    pca_data = pca_data.merge(
        cluster_data,
        on='AH',
        how='left',
        validate='one_to_one',
    )

    if pca_data['cluster_id'].isna().any():
        raise ValueError('Some hydrological years do not have cluster labels.')

    explained_variance = pca.explained_variance_ratio_.tolist()

    return pca_data.sort_values('AH_YEAR').reset_index(drop=True), explained_variance


def plot_selected_grouping_pca_projection(
    feature_matrix: pd.DataFrame,
    membership: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plot PCA projection of the selected grouping."""
    data, explained_variance = prepare_selected_grouping_pca_data(
        feature_matrix=feature_matrix,
        membership=membership,
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    for cluster_id, cluster_data in data.groupby('cluster_id', sort=True):
        ax.scatter(
            cluster_data['PC1'],
            cluster_data['PC2'],
            s=46,
            label=format_cluster_label(cluster_id),
            zorder=3,
        )

        for _, row in cluster_data.iterrows():
            label = str(row['AH']).replace('AH-', '')
            ax.annotate(
                label,
                (row['PC1'], row['PC2']),
                textcoords='offset points',
                xytext=(0, 6),
                ha='center',
                fontsize=7,
            )

    ax.axhline(0, linewidth=0.7, alpha=0.35)
    ax.axvline(0, linewidth=0.7, alpha=0.35)

    ax.set_xlabel(
        f'PC1 ({explained_variance[0] * 100:.1f}%)',
        fontsize=8,
    )
    ax.set_ylabel(
        f'PC2 ({explained_variance[1] * 100:.1f}%)',
        fontsize=8,
    )
    ax.tick_params(axis='both', labelsize=7)
    ax.grid(True, linewidth=0.4, alpha=0.35)

    ax.legend(
        title='Cluster',
        fontsize=7,
        title_fontsize=7,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.18),
        ncol=data['cluster_id'].nunique(),
        frameon=True,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return output_path


def plot_feature_correlation_heatmap(
    correlation_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plot feature correlation heatmap."""
    matrix = prepare_feature_correlation_matrix(correlation_df)

    fig, ax = plt.subplots(figsize=(6.8, 5.4))

    image = ax.imshow(
        matrix.values,
        cmap='coolwarm',
        vmin=-1.0,
        vmax=1.0,
        aspect='auto',
    )

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticklabels(
        matrix.columns,
        rotation=45,
        ha='right',
        fontsize=7,
    )
    ax.set_yticklabels(matrix.index, fontsize=7)

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix.iloc[row_index, col_index]
            text_color = 'white' if abs(value) >= 0.50 else 'black'
            ax.text(
                col_index,
                row_index,
                f'{value:.2f}',
                ha='center',
                va='center',
                color=text_color,
                fontsize=8,
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label('Pearson correlation', fontsize=8)
    colorbar.ax.tick_params(labelsize=7)

    ax.tick_params(axis='both', length=0)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return output_path

def plot_final_partition_scores(
    final_partitions: pd.DataFrame,
    output_path: Path,
    top_n: int = 10,
) -> Path:
    """Plot final scores for top-ranked groupings."""
    data = prepare_partition_score_data(final_partitions, top_n=top_n)
    labels = (
        data["grouping_label"].astype(str)
        + "\nK="
        + data["n_clusters"].astype(str)
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, data["final_score"])

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Candidate grouping", fontsize=8)
    ax.set_ylabel("Final integrated score", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linewidth=0.4, alpha=0.35)

    for bar, score in zip(bars, data["final_score"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_cluster_difference_profile(
    differences: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plot standardized cluster differences against the global mean."""
    matrix = prepare_cluster_difference_matrix(differences)

    fig, ax = plt.subplots(figsize=(8, 4))
    matrix.plot(kind="bar", ax=ax)

    ax.axhline(0, linewidth=1, color="#6c757d")
    ax.set_xlabel("Feature", fontsize=8)
    ax.set_ylabel("Standardized deviation from global mean", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linewidth=0.4, alpha=0.35)
    ax.legend(
        title="Cluster",
        fontsize=7,
        title_fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.32),
        ncol=max(1, len(matrix.columns)),
        frameon=True,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_year_typology(
    membership: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plot annual hydrological-operational typology."""
    data = prepare_year_typology_data(membership)
    cluster_order = sorted(data["cluster_id"].astype(str).unique())

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.scatter(
        data["AH_YEAR"],
        data["cluster_position"],
        s=42,
        zorder=3,
    )

    for _, row in data.iterrows():
        label = str(row["AH"]).replace("AH-", "")
        ax.annotate(
            label,
            (row["AH_YEAR"], row["cluster_position"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=7,
        )

    ax.set_yticks(range(len(cluster_order)))
    ax.set_yticklabels(cluster_order)
    ax.set_xlabel("Hydrological-operational year", fontsize=8)
    ax.set_ylabel("Selected cluster", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, linewidth=0.4, alpha=0.35)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_best_score_by_algorithm(
    algorithm_summary: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plot best final score reached by each algorithm."""
    data = prepare_algorithm_score_data(algorithm_summary)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(data["algorithm_label"], data["best_final_score"])

    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Best final integrated score", fontsize=8)
    ax.set_ylabel("Algorithm", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.invert_yaxis()
    ax.grid(axis="x", linewidth=0.4, alpha=0.35)

    for bar, (_, row) in zip(bars, data.iterrows()):
        label = (
            f"{row['best_final_score']:.3f} | "
            f"K={int(row['best_n_clusters'])}"
        )
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=7,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path

def plot_valid_partitions_by_algorithm(
    algorithm_summary: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plot retained grouping counts by algorithm."""
    data = prepare_valid_partition_data(algorithm_summary)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(data["algorithm_label"], data["n_valid_partitions"])

    max_configurations = int(data["n_configurations"].max())
    ax.set_xlim(0, max_configurations + 2)
    ax.set_xlabel("Retained groupings", fontsize=8)
    ax.set_ylabel("Algorithm", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.invert_yaxis()
    ax.grid(axis="x", linewidth=0.4, alpha=0.35)

    for bar, (_, row) in zip(bars, data.iterrows()):
        rate = row["valid_partition_rate"] * 100
        label = (
            f"{int(row['n_valid_partitions'])}/"
            f"{int(row['n_configurations'])} ({rate:.0f}%)"
        )
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=7,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def get_figure_paths(
    experiments_config: dict[str, Any],
) -> dict[str, Path]:
    """Return output paths for publication result figures."""
    figure_dir = get_figure_output_dir(experiments_config)

    return {
        'valid_partitions_by_algorithm': (
            figure_dir / 'valid_partitions_by_algorithm.png'
        ),
        'feature_correlation_heatmap': (
            figure_dir / 'feature_correlation_heatmap.png'
        ),
        'selected_grouping_pca_projection': (
            figure_dir / 'selected_grouping_pca_projection.png'
        ),
    }


def generate_figures(
    experiments_config: dict[str, Any],
) -> dict[str, Path]:
    """Generate publication result figures."""
    (
        _final_partitions,
        _differences,
        membership,
        algorithm_summary,
    ) = load_required_outputs(experiments_config)

    correlation_df = load_feature_correlation_data(experiments_config)
    feature_matrix = load_standardized_feature_matrix(experiments_config)
    figure_paths = get_figure_paths(experiments_config)

    plot_valid_partitions_by_algorithm(
        algorithm_summary=algorithm_summary,
        output_path=figure_paths['valid_partitions_by_algorithm'],
    )
    plot_feature_correlation_heatmap(
        correlation_df=correlation_df,
        output_path=figure_paths['feature_correlation_heatmap'],
    )
    plot_selected_grouping_pca_projection(
        feature_matrix=feature_matrix,
        membership=membership,
        output_path=figure_paths['selected_grouping_pca_projection'],
    )

    print('[figures] Figuras de resultados generadas correctamente')
    for name, path in figure_paths.items():
        print(f'[figures] {name}: {path}')

    return figure_paths


if __name__ == '__main__':
    from src.config_loader import load_experiments_config

    experiments_config = load_experiments_config()
    generate_figures(experiments_config)