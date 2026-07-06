from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config_loader import load_experiments_config, load_pipeline_config
from src.partition_equivalence import canonicalize_labels


METADATA_COLUMNS = {
    'AH',
    'AH_YEAR',
    'run_id',
    'algorithm_name',
    'feature_set_name',
    'scaler_name',
    'cluster_label',
    'canonical_cluster',
    'cluster_id',
    'selected_partition_id',
    'representative_run_id',
    'representative_algorithm',
}


def get_output_dir(experiments_config: dict[str, Any]) -> Path:
    """Return benchmark output table directory."""
    benchmark_path = Path(
        experiments_config['salidas']['tablas']['benchmark_runs']
    )

    return benchmark_path.parent


def get_features_path(pipeline_config: dict[str, Any]) -> Path:
    """Return annual feature matrix path with safe fallbacks."""
    candidates = [
        ('paths', 'features'),
        ('paths', 'features_path'),
        ('paths', 'processed_features'),
        ('rutas', 'features'),
        ('rutas', 'features_path'),
        ('rutas', 'features_anuales'),
    ]

    for section, key in candidates:
        value = pipeline_config.get(section, {}).get(key)
        if value is not None:
            return Path(value)

    return Path('data/processed/features.csv')


def get_feature_columns(data: pd.DataFrame) -> list[str]:
    """Return numeric feature columns excluding metadata."""
    return [
        column
        for column in data.columns
        if column not in METADATA_COLUMNS
        and pd.api.types.is_numeric_dtype(data[column])
    ]


def select_representative_partition(
    final_partitions: pd.DataFrame,
    final_partition_rank: int = 1,
) -> pd.Series:
    """Select the partition to interpret."""
    if final_partitions.empty:
        raise ValueError('Final partition ranking is empty.')

    if 'final_partition_rank' in final_partitions.columns:
        selected = final_partitions[
            final_partitions['final_partition_rank'] == final_partition_rank
        ]

        if selected.empty:
            raise ValueError(
                f'No partition found with rank {final_partition_rank}.'
            )

        return selected.iloc[0]

    return final_partitions.sort_values(
        by='final_score',
        ascending=False,
    ).iloc[0]


def get_representative_labels(
    labels: pd.DataFrame,
    representative_run_id: str,
) -> pd.DataFrame:
    """Return labels from the representative run with canonical clusters."""
    selected = labels[
        labels['run_id'].astype(str) == str(representative_run_id)
    ].copy()

    if selected.empty:
        raise ValueError(
            f'No labels found for run_id: {representative_run_id}'
        )

    selected = selected.sort_values('AH_YEAR').reset_index(drop=True)
    canonical = canonicalize_labels(
        selected['cluster_label'].astype(int).tolist()
    )

    selected['canonical_cluster'] = list(canonical)
    selected['cluster_id'] = selected['canonical_cluster'].apply(
        lambda value: 'noise' if value == -1 else f'cluster_{value + 1}'
    )

    return selected


def build_selected_partition_membership(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    final_partitions: pd.DataFrame,
    final_partition_rank: int = 1,
) -> pd.DataFrame:
    """Build annual membership table for the selected partition."""
    selected_partition = select_representative_partition(
        final_partitions,
        final_partition_rank,
    )

    representative_run_id = str(selected_partition['representative_run_id'])
    representative_algorithm = str(
        selected_partition['representative_algorithm']
    )
    partition_id = str(selected_partition['partition_id'])

    selected_labels = get_representative_labels(
        labels,
        representative_run_id,
    )

    merged = selected_labels.merge(
        features,
        on=['AH', 'AH_YEAR'],
        how='left',
    )

    feature_columns = get_feature_columns(features)

    if merged[feature_columns].isna().any().any():
        raise ValueError('Some selected labels do not match feature rows.')

    merged.insert(0, 'selected_partition_id', partition_id)
    merged.insert(1, 'representative_run_id', representative_run_id)
    merged.insert(2, 'representative_algorithm', representative_algorithm)

    ordered_columns = [
        'selected_partition_id',
        'representative_run_id',
        'representative_algorithm',
        'AH',
        'AH_YEAR',
        'cluster_label',
        'canonical_cluster',
        'cluster_id',
        *feature_columns,
    ]

    return merged[ordered_columns].sort_values('AH_YEAR').reset_index(
        drop=True,
    )


def build_cluster_profiles(
    membership: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Summarize feature means by selected cluster."""
    rows: list[dict[str, Any]] = []

    for cluster_id, group in membership.groupby('cluster_id', sort=True):
        row: dict[str, Any] = {
            'cluster_id': cluster_id,
            'n_years': len(group),
            'years': ', '.join(group['AH'].astype(str).tolist()),
        }

        for feature in feature_columns:
            row[f'{feature}_mean'] = float(group[feature].mean())

        rows.append(row)

    return pd.DataFrame(rows)


def build_cluster_feature_differences(
    membership: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Compare each cluster mean against the global mean."""
    overall_mean = membership[feature_columns].mean()
    overall_std = membership[feature_columns].std(ddof=0).replace(0, np.nan)

    rows: list[dict[str, Any]] = []

    for cluster_id, group in membership.groupby('cluster_id', sort=True):
        cluster_mean = group[feature_columns].mean()

        for feature in feature_columns:
            difference = float(cluster_mean[feature] - overall_mean[feature])
            std_value = overall_std[feature]

            standardized_difference = (
                float(difference / std_value)
                if np.isfinite(std_value) and std_value != 0
                else float('nan')
            )

            if np.isclose(difference, 0.0):
                direction = 'similar'
            elif difference > 0:
                direction = 'above_global_mean'
            else:
                direction = 'below_global_mean'

            rows.append(
                {
                    'cluster_id': cluster_id,
                    'feature': feature,
                    'cluster_mean': float(cluster_mean[feature]),
                    'global_mean': float(overall_mean[feature]),
                    'difference': difference,
                    'standardized_difference': standardized_difference,
                    'direction': direction,
                }
            )

    differences = pd.DataFrame(rows)

    return differences.sort_values(
        by='standardized_difference',
        key=lambda values: values.abs(),
        ascending=False,
    ).reset_index(drop=True)


def get_interpretation_output_paths(
    experiments_config: dict[str, Any],
) -> tuple[Path, Path, Path]:
    """Return interpretation output paths."""
    output_dir = get_output_dir(experiments_config)

    return (
        output_dir / 'selected_partition_membership.csv',
        output_dir / 'selected_cluster_profiles.csv',
        output_dir / 'selected_cluster_feature_differences.csv',
    )


def save_interpretation_outputs(
    membership: pd.DataFrame,
    profiles: pd.DataFrame,
    differences: pd.DataFrame,
    experiments_config: dict[str, Any],
) -> tuple[Path, Path, Path]:
    """Save interpretation tables."""
    membership_path, profiles_path, differences_path = (
        get_interpretation_output_paths(experiments_config)
    )

    membership_path.parent.mkdir(parents=True, exist_ok=True)

    membership.to_csv(membership_path, index=False, encoding='utf-8')
    profiles.to_csv(profiles_path, index=False, encoding='utf-8')
    differences.to_csv(differences_path, index=False, encoding='utf-8')

    return membership_path, profiles_path, differences_path


def load_required_outputs(
    pipeline_config: dict[str, Any],
    experiments_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load features, labels and final partition ranking."""
    output_dir = get_output_dir(experiments_config)

    features = pd.read_csv(get_features_path(pipeline_config))
    labels = pd.read_csv(output_dir / 'benchmark_labels.csv')
    final_partitions = pd.read_csv(output_dir / 'final_partition_ranking.csv')

    return features, labels, final_partitions


def run_interpretation(
    pipeline_config: dict[str, Any],
    experiments_config: dict[str, Any],
    final_partition_rank: int = 1,
) -> pd.DataFrame:
    """Run interpretation for the selected final partition."""
    features, labels, final_partitions = load_required_outputs(
        pipeline_config,
        experiments_config,
    )

    membership = build_selected_partition_membership(
        features=features,
        labels=labels,
        final_partitions=final_partitions,
        final_partition_rank=final_partition_rank,
    )
    feature_columns = get_feature_columns(features)

    profiles = build_cluster_profiles(membership, feature_columns)
    differences = build_cluster_feature_differences(
        membership,
        feature_columns,
    )

    membership_path, profiles_path, differences_path = (
        save_interpretation_outputs(
            membership=membership,
            profiles=profiles,
            differences=differences,
            experiments_config=experiments_config,
        )
    )

    selected_partition = select_representative_partition(
        final_partitions,
        final_partition_rank,
    )

    print('[interpretation] Interpretación de partición seleccionada')
    print(f"[interpretation] Partición: {selected_partition['partition_id']}")
    print(
        '[interpretation] Run representante: '
        f"{selected_partition['representative_run_id']}"
    )
    print(
        '[interpretation] Algoritmo representante: '
        f"{selected_partition['representative_algorithm']}"
    )
    print(f'[interpretation] Tabla membresía: {membership_path}')
    print(f'[interpretation] Tabla perfiles: {profiles_path}')
    print(f'[interpretation] Tabla diferencias: {differences_path}')

    print('[interpretation] Años por clúster:')
    years_by_cluster = membership.groupby('cluster_id')['AH'].apply(
        lambda values: ', '.join(values.astype(str))
    )
    print(years_by_cluster.to_string())

    print('[interpretation] Perfiles promedio por clúster:')
    profile_columns = ['cluster_id', 'n_years']
    profile_columns.extend(f'{feature}_mean' for feature in feature_columns)
    print(profiles[profile_columns].round(3).to_string(index=False))

    print('[interpretation] Diferencias más relevantes:')
    print(differences.head(10).round(3).to_string(index=False))

    return membership


if __name__ == '__main__':
    pipeline_config = load_pipeline_config()
    experiments_config = load_experiments_config()

    run_interpretation(
        pipeline_config=pipeline_config,
        experiments_config=experiments_config,
    )
