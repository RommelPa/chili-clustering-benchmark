from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config_loader import load_experiments_config, load_pipeline_config
from src.features import generate_annual_features


@dataclass(frozen=True)
class ExperimentMatrix:
    """Container for a scaled feature matrix used in clustering experiments."""

    feature_set_name: str
    scaler_name: str
    variables: list[str]
    ah_labels: list[str]
    ah_years: list[int]
    values: np.ndarray
    scaled_frame: pd.DataFrame


def get_feature_sets(experiments_config: dict[str, Any]) -> dict[str, list[str]]:
    """Return configured feature sets for benchmarking."""
    feature_sets = experiments_config['preprocesamiento_experimental']['feature_sets']

    return {
        name: spec['variables']
        for name, spec in feature_sets.items()
    }


def get_scalers(experiments_config: dict[str, Any]) -> list[str]:
    """Return configured scalers for benchmarking."""
    return experiments_config['preprocesamiento_experimental']['scalers']


def validate_feature_sets(
    features: pd.DataFrame,
    experiments_config: dict[str, Any],
) -> None:
    """Validate that all configured feature sets exist in the feature matrix."""
    available_features = set(features.columns)
    feature_sets = get_feature_sets(experiments_config)

    for feature_set_name, variables in feature_sets.items():
        missing = [
            variable for variable in variables
            if variable not in available_features
        ]

        if missing:
            missing_text = ', '.join(missing)
            raise ValueError(
                f'Missing variables in feature set {feature_set_name}: '
                f'{missing_text}'
            )


def scale_feature_values(
    features: pd.DataFrame,
    variables: list[str],
    scaler_name: str,
) -> np.ndarray:
    """Scale selected feature columns according to the configured scaler."""
    values = features[variables].to_numpy(dtype=float)

    if scaler_name == 'zscore':
        scaler = StandardScaler()
        return scaler.fit_transform(values)

    raise ValueError(f'Unsupported scaler: {scaler_name}')


def build_scaled_frame(
    features: pd.DataFrame,
    variables: list[str],
    scaled_values: np.ndarray,
) -> pd.DataFrame:
    """Build a DataFrame with AH metadata and scaled feature values."""
    scaled = pd.DataFrame(
        scaled_values,
        columns=variables,
        index=features.index,
    )

    return pd.concat(
        [
            features[['AH', 'AH_YEAR']].reset_index(drop=True),
            scaled.reset_index(drop=True),
        ],
        axis=1,
    )


def build_experiment_matrices(
    features: pd.DataFrame,
    experiments_config: dict[str, Any],
) -> list[ExperimentMatrix]:
    """Build all configured experimental matrices."""
    validate_feature_sets(features, experiments_config)

    matrices: list[ExperimentMatrix] = []
    feature_sets = get_feature_sets(experiments_config)
    scalers = get_scalers(experiments_config)

    for feature_set_name, variables in feature_sets.items():
        for scaler_name in scalers:
            scaled_values = scale_feature_values(
                features=features,
                variables=variables,
                scaler_name=scaler_name,
            )
            scaled_frame = build_scaled_frame(
                features=features,
                variables=variables,
                scaled_values=scaled_values,
            )

            matrices.append(
                ExperimentMatrix(
                    feature_set_name=feature_set_name,
                    scaler_name=scaler_name,
                    variables=variables,
                    ah_labels=features['AH'].tolist(),
                    ah_years=features['AH_YEAR'].astype(int).tolist(),
                    values=scaled_values,
                    scaled_frame=scaled_frame,
                )
            )

    return matrices


def save_experiment_matrices(
    matrices: list[ExperimentMatrix],
    output_dir: str | Path,
) -> list[Path]:
    """Save scaled experimental matrices as CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []

    for matrix in matrices:
        file_name = (
            f'matrix_{matrix.feature_set_name}_{matrix.scaler_name}.csv'
        )
        path = output_path / file_name
        matrix.scaled_frame.to_csv(path, index=False, encoding='utf-8')
        saved_paths.append(path)

    return saved_paths


def prepare_experiment_matrices(
    pipeline_config: dict[str, Any],
    experiments_config: dict[str, Any],
) -> list[ExperimentMatrix]:
    """Generate annual features and prepare scaled matrices for experiments."""
    features = generate_annual_features(pipeline_config)
    matrices = build_experiment_matrices(features, experiments_config)

    output_dir = pipeline_config['rutas']['carpeta_tablas']
    save_experiment_matrices(matrices, output_dir)

    return matrices


if __name__ == '__main__':
    pipeline_config = load_pipeline_config()
    experiments_config = load_experiments_config()

    experiment_matrices = prepare_experiment_matrices(
        pipeline_config=pipeline_config,
        experiments_config=experiments_config,
    )

    print('[experiment_matrix] Matrices experimentales preparadas')
    print(f'[experiment_matrix] Total de matrices: {len(experiment_matrices)}')

    for matrix in experiment_matrices:
        print(
            '[experiment_matrix] '
            f'{matrix.feature_set_name} | '
            f'{matrix.scaler_name} | '
            f'{matrix.values.shape[0]} años x {matrix.values.shape[1]} features'
        )

