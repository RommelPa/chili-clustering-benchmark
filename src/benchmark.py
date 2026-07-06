from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from src.clustering_models import ClusteringRunSpec, build_run_specs, fit_predict_clustering
from src.config_loader import load_experiments_config, load_pipeline_config
from src.experiment_matrix import ExperimentMatrix, prepare_experiment_matrices
from src.metrics import build_metrics_row


def configure_mlflow(experiments_config: dict[str, Any]) -> None:
    """Configure local MLflow tracking for benchmark runs."""
    mlflow_config = experiments_config['mlflow']
    tracking_uri = mlflow_config['tracking_uri']

    if tracking_uri.startswith('sqlite:///'):
        database_path = Path(tracking_uri.replace('sqlite:///', ''))
        database_path.parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_config['experiment_name'])


def is_loggable_number(value: Any) -> bool:
    """Return True when a value can be safely logged as an MLflow metric."""
    if isinstance(value, bool):
        return False

    if isinstance(value, int | float | np.integer | np.floating):
        return bool(np.isfinite(float(value)))

    return False


def build_failed_metrics_row(
    spec: ClusteringRunSpec,
    matrix: ExperimentMatrix,
    error: Exception,
) -> dict[str, Any]:
    """Build a metrics row for a failed clustering run."""
    row: dict[str, Any] = {
        'run_id': spec.run_id,
        'algorithm_name': spec.algorithm_name,
        'family': spec.family,
        'role': spec.role,
        'feature_set_name': matrix.feature_set_name,
        'scaler_name': matrix.scaler_name,
        'n_clusters': 0,
        'n_noise': 0,
        'noise_ratio': 0.0,
        'cluster_min_size': 0,
        'cluster_max_size': 0,
        'silhouette': float('nan'),
        'davies_bouldin': float('nan'),
        'calinski_harabasz': float('nan'),
        'is_valid_partition': False,
        'status': 'failed',
        'error_message': str(error),
    }

    for param_name, param_value in spec.params.items():
        row[f'param_{param_name}'] = param_value

    return row


def build_labels_frame(
    spec: ClusteringRunSpec,
    matrix: ExperimentMatrix,
    labels: np.ndarray,
) -> pd.DataFrame:
    """Build a flat labels DataFrame for one clustering run."""
    return pd.DataFrame(
        {
            'run_id': spec.run_id,
            'algorithm_name': spec.algorithm_name,
            'feature_set_name': matrix.feature_set_name,
            'scaler_name': matrix.scaler_name,
            'AH': matrix.ah_labels,
            'AH_YEAR': matrix.ah_years,
            'cluster_label': labels.astype(int),
        }
    )


def run_single_benchmark(
    spec: ClusteringRunSpec,
    matrix: ExperimentMatrix,
    restrictions: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    """Execute one clustering configuration and return metrics and labels."""
    try:
        labels = fit_predict_clustering(spec, matrix.values)
        row = build_metrics_row(
            run_id=spec.run_id,
            algorithm_name=spec.algorithm_name,
            family=spec.family,
            role=spec.role,
            feature_set_name=matrix.feature_set_name,
            scaler_name=matrix.scaler_name,
            params=spec.params,
            values=matrix.values,
            labels=labels,
            restrictions=restrictions,
        )
        row['status'] = 'success'
        row['error_message'] = ''

        labels_frame = build_labels_frame(spec, matrix, labels)

        return row, labels_frame

    except Exception as error:
        return build_failed_metrics_row(spec, matrix, error), None


def log_run_to_mlflow(
    row: dict[str, Any],
    spec: ClusteringRunSpec,
    matrix: ExperimentMatrix,
) -> None:
    """Log one benchmark run to MLflow."""
    run_name = (
        f'{row["run_id"]}_'
        f'{matrix.feature_set_name}_'
        f'{matrix.scaler_name}'
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag('algorithm_name', spec.algorithm_name)
        mlflow.set_tag('family', spec.family)
        mlflow.set_tag('role', spec.role)
        mlflow.set_tag('feature_set_name', matrix.feature_set_name)
        mlflow.set_tag('scaler_name', matrix.scaler_name)
        mlflow.set_tag('status', row['status'])

        if row.get('error_message'):
            mlflow.set_tag('error_message', row['error_message'])

        for param_name, param_value in spec.params.items():
            mlflow.log_param(param_name, param_value)

        mlflow.log_param('n_features', len(matrix.variables))
        mlflow.log_param('n_years', len(matrix.ah_labels))

        for metric_name, metric_value in row.items():
            if is_loggable_number(metric_value):
                mlflow.log_metric(metric_name, float(metric_value))

        mlflow.log_metric(
            'is_valid_partition',
            int(bool(row['is_valid_partition'])),
        )


def save_benchmark_outputs(
    rows: list[dict[str, Any]],
    labels_frames: list[pd.DataFrame],
    experiments_config: dict[str, Any],
) -> tuple[Path, Path]:
    """Save benchmark metrics and labels tables."""
    benchmark_path = Path(
        experiments_config['salidas']['tablas']['benchmark_runs']
    )
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)

    benchmark = pd.DataFrame(rows)
    benchmark.to_csv(benchmark_path, index=False, encoding='utf-8')

    labels_path = benchmark_path.parent / 'benchmark_labels.csv'

    if labels_frames:
        labels = pd.concat(labels_frames, ignore_index=True)
    else:
        labels = pd.DataFrame()

    labels.to_csv(labels_path, index=False, encoding='utf-8')

    return benchmark_path, labels_path


def run_benchmark(
    pipeline_config: dict[str, Any],
    experiments_config: dict[str, Any],
    log_to_mlflow: bool = True,
) -> pd.DataFrame:
    """Run the full clustering benchmark."""
    matrices = prepare_experiment_matrices(
        pipeline_config=pipeline_config,
        experiments_config=experiments_config,
    )
    specs = build_run_specs(experiments_config)
    restrictions = experiments_config['restricciones_particion']

    if log_to_mlflow:
        configure_mlflow(experiments_config)

    rows: list[dict[str, Any]] = []
    labels_frames: list[pd.DataFrame] = []

    for matrix in matrices:
        for spec in specs:
            row, labels_frame = run_single_benchmark(
                spec=spec,
                matrix=matrix,
                restrictions=restrictions,
            )

            rows.append(row)

            if labels_frame is not None:
                labels_frames.append(labels_frame)

            if log_to_mlflow:
                log_run_to_mlflow(row, spec, matrix)

    benchmark_path, labels_path = save_benchmark_outputs(
        rows=rows,
        labels_frames=labels_frames,
        experiments_config=experiments_config,
    )

    benchmark = pd.DataFrame(rows)

    print('[benchmark] Benchmark ejecutado correctamente')
    print(f'[benchmark] Configuraciones evaluadas: {len(benchmark)}')
    print(f'[benchmark] Correctas: {(benchmark["status"] == "success").sum()}')
    print(f'[benchmark] Fallidas: {(benchmark["status"] == "failed").sum()}')
    print(
        '[benchmark] Particiones válidas: '
        f'{benchmark["is_valid_partition"].sum()}'
    )
    print(f'[benchmark] Tabla de métricas: {benchmark_path}')
    print(f'[benchmark] Tabla de etiquetas: {labels_path}')

    return benchmark


if __name__ == '__main__':
    pipeline_config = load_pipeline_config()
    experiments_config = load_experiments_config()

    run_benchmark(
        pipeline_config=pipeline_config,
        experiments_config=experiments_config,
        log_to_mlflow=True,
    )


