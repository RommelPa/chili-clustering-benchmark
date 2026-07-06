from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.clustering_models import ClusteringRunSpec, build_run_specs, fit_predict_clustering
from src.config_loader import load_experiments_config, load_pipeline_config
from src.experiment_matrix import ExperimentMatrix, prepare_experiment_matrices


def get_sample_size(n_items: int, sample_fraction: float) -> int:
    """Calculate subsample size from the configured fraction."""
    if n_items < 2:
        return n_items

    sample_size = int(round(n_items * sample_fraction))

    return max(2, min(n_items, sample_size))


def calculate_pairwise_stability(
    reference_labels: np.ndarray,
    candidate_labels: np.ndarray,
) -> dict[str, float]:
    """Compare two partitions using pairwise co-association agreement."""
    if len(reference_labels) != len(candidate_labels):
        raise ValueError('Reference and candidate labels must have same length.')

    n_items = len(reference_labels)

    if n_items < 2:
        return {
            'stability_intra': float('nan'),
            'stability_inter': float('nan'),
            'stability_score': float('nan'),
            'stability_gap': float('nan'),
        }

    reference = reference_labels.astype(int)
    candidate = candidate_labels.astype(int)

    upper_mask = np.triu(np.ones((n_items, n_items), dtype=bool), k=1)

    reference_non_noise = reference != -1
    valid_reference_pairs = (
        upper_mask
        & reference_non_noise[:, None]
        & reference_non_noise[None, :]
    )

    reference_same = (
        (reference[:, None] == reference[None, :])
        & reference_non_noise[:, None]
        & reference_non_noise[None, :]
    )

    candidate_same = (
        (candidate[:, None] == candidate[None, :])
        & (candidate[:, None] != -1)
        & (candidate[None, :] != -1)
    )

    intra_mask = valid_reference_pairs & reference_same
    inter_mask = valid_reference_pairs & ~reference_same

    stability_intra = (
        float(candidate_same[intra_mask].mean())
        if intra_mask.any()
        else float('nan')
    )
    stability_inter = (
        float((~candidate_same[inter_mask]).mean())
        if inter_mask.any()
        else float('nan')
    )

    valid_scores = [
        value
        for value in [stability_intra, stability_inter]
        if np.isfinite(value)
    ]

    stability_score = (
        float(np.mean(valid_scores))
        if valid_scores
        else float('nan')
    )

    stability_gap = (
        float(abs(stability_intra - stability_inter))
        if np.isfinite(stability_intra) and np.isfinite(stability_inter)
        else float('nan')
    )

    return {
        'stability_intra': stability_intra,
        'stability_inter': stability_inter,
        'stability_score': stability_score,
        'stability_gap': stability_gap,
    }


def get_reference_labels(
    labels: pd.DataFrame,
    run_id: str,
) -> np.ndarray:
    """Return reference labels sorted by hydrological year."""
    run_labels = labels[labels['run_id'].astype(str) == str(run_id)].copy()

    if run_labels.empty:
        raise ValueError(f'No labels found for run_id: {run_id}')

    ordered = run_labels.sort_values('AH_YEAR')

    return ordered['cluster_label'].astype(int).to_numpy()


def summarize_stability_iterations(
    iteration_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize stability iteration results."""
    iterations = pd.DataFrame(iteration_rows)
    successful = iterations[iterations['status'] == 'success']

    summary: dict[str, Any] = {
        'n_iterations': len(iterations),
        'n_successful_iterations': len(successful),
        'n_failed_iterations': int((iterations['status'] == 'failed').sum()),
    }

    for column in [
        'stability_intra',
        'stability_inter',
        'stability_score',
        'stability_gap',
    ]:
        summary[column] = (
            float(successful[column].mean())
            if not successful.empty
            else float('nan')
        )

    return summary


def compute_run_stability(
    spec: ClusteringRunSpec,
    matrix: ExperimentMatrix,
    reference_labels: np.ndarray,
    n_iterations: int,
    sample_fraction: float,
    random_state: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Estimate clustering stability for one benchmark run."""
    rng = np.random.default_rng(random_state)
    n_items = len(reference_labels)
    sample_size = get_sample_size(n_items, sample_fraction)

    iteration_rows: list[dict[str, Any]] = []

    for iteration in range(1, n_iterations + 1):
        sample_indices = np.sort(
            rng.choice(
                n_items,
                size=sample_size,
                replace=False,
            )
        )

        try:
            sampled_values = matrix.values[sample_indices]
            sampled_reference = reference_labels[sample_indices]
            sampled_labels = fit_predict_clustering(spec, sampled_values)

            stability = calculate_pairwise_stability(
                reference_labels=sampled_reference,
                candidate_labels=sampled_labels,
            )

            iteration_rows.append(
                {
                    'iteration': iteration,
                    'run_id': spec.run_id,
                    'status': 'success',
                    'sample_size': sample_size,
                    **stability,
                    'error_message': '',
                }
            )

        except Exception as error:
            iteration_rows.append(
                {
                    'iteration': iteration,
                    'run_id': spec.run_id,
                    'status': 'failed',
                    'sample_size': sample_size,
                    'stability_intra': float('nan'),
                    'stability_inter': float('nan'),
                    'stability_score': float('nan'),
                    'stability_gap': float('nan'),
                    'error_message': str(error),
                }
            )

    summary = summarize_stability_iterations(iteration_rows)

    return summary, pd.DataFrame(iteration_rows)


def build_spec_map(
    experiments_config: dict[str, Any],
) -> dict[str, ClusteringRunSpec]:
    """Build a mapping from run_id to clustering run specification."""
    specs = build_run_specs(experiments_config)

    return {spec.run_id: spec for spec in specs}


def build_matrix_map(
    matrices: list[ExperimentMatrix],
) -> dict[tuple[str, str], ExperimentMatrix]:
    """Build a mapping from feature set and scaler to matrix."""
    return {
        (matrix.feature_set_name, matrix.scaler_name): matrix
        for matrix in matrices
    }


def get_stability_output_paths(
    experiments_config: dict[str, Any],
) -> tuple[Path, Path]:
    """Return output paths for stability tables."""
    benchmark_path = Path(
        experiments_config['salidas']['tablas']['benchmark_runs']
    )
    output_dir = benchmark_path.parent

    return (
        output_dir / 'stability_runs.csv',
        output_dir / 'stability_iterations.csv',
    )


def get_project_random_state(pipeline_config: dict[str, Any]) -> int:
    """Return configured project random state for reproducibility."""
    project_config = pipeline_config['proyecto']

    return int(
        project_config.get(
            'random_state',
            project_config.get('seed', 42),
        )
    )

def save_stability_outputs(
    stability_rows: list[dict[str, Any]],
    iteration_frames: list[pd.DataFrame],
    experiments_config: dict[str, Any],
) -> tuple[Path, Path]:
    """Save stability summary and iteration-level results."""
    stability_path, iterations_path = get_stability_output_paths(
        experiments_config,
    )
    stability_path.parent.mkdir(parents=True, exist_ok=True)

    stability = pd.DataFrame(stability_rows)
    stability.to_csv(stability_path, index=False, encoding='utf-8')

    if iteration_frames:
        iterations = pd.concat(iteration_frames, ignore_index=True)
    else:
        iterations = pd.DataFrame()

    iterations.to_csv(iterations_path, index=False, encoding='utf-8')

    return stability_path, iterations_path


def run_stability_analysis(
    pipeline_config: dict[str, Any],
    experiments_config: dict[str, Any],
    max_runs: int | None = None,
) -> pd.DataFrame:
    """Run subsampling stability analysis for ranked benchmark runs."""
    benchmark_path = Path(
        experiments_config['salidas']['tablas']['benchmark_runs']
    )
    output_dir = benchmark_path.parent

    ranking = pd.read_csv(output_dir / 'ranking_internal.csv')
    labels = pd.read_csv(output_dir / 'benchmark_labels.csv')

    if max_runs is not None:
        ranking = ranking.head(max_runs).copy()

    matrices = prepare_experiment_matrices(
        pipeline_config=pipeline_config,
        experiments_config=experiments_config,
    )
    spec_map = build_spec_map(experiments_config)
    matrix_map = build_matrix_map(matrices)

    stability_config = experiments_config['estabilidad']
    n_iterations = int(stability_config['n_iteraciones'])
    sample_fraction = float(stability_config['fraccion_muestra'])
    random_state = get_project_random_state(pipeline_config)

    stability_rows: list[dict[str, Any]] = []
    iteration_frames: list[pd.DataFrame] = []

    for index, row in ranking.iterrows():
        spec = spec_map[str(row['run_id'])]
        matrix_key = (row['feature_set_name'], row['scaler_name'])
        matrix = matrix_map[matrix_key]
        reference_labels = get_reference_labels(labels, spec.run_id)

        summary, iterations = compute_run_stability(
            spec=spec,
            matrix=matrix,
            reference_labels=reference_labels,
            n_iterations=n_iterations,
            sample_fraction=sample_fraction,
            random_state=random_state + int(index),
        )

        stability_rows.append(
            {
                'rank': int(row['rank']),
                'run_id': row['run_id'],
                'algorithm_name': row['algorithm_name'],
                'feature_set_name': row['feature_set_name'],
                'scaler_name': row['scaler_name'],
                'ranking_score': float(row['ranking_score']),
                'n_clusters': int(row['n_clusters']),
                **summary,
            }
        )
        iteration_frames.append(iterations)

    stability_path, iterations_path = save_stability_outputs(
        stability_rows=stability_rows,
        iteration_frames=iteration_frames,
        experiments_config=experiments_config,
    )

    stability = pd.DataFrame(stability_rows).sort_values(
        by=['stability_score', 'ranking_score'],
        ascending=[False, False],
    ).reset_index(drop=True)

    print('[stability] Estabilidad por submuestreo calculada')
    print(f'[stability] Corridas evaluadas: {len(stability)}')
    print(f'[stability] Iteraciones por corrida: {n_iterations}')
    print(f'[stability] Fracción de muestra: {sample_fraction}')
    print(f'[stability] Tabla resumen: {stability_path}')
    print(f'[stability] Tabla iteraciones: {iterations_path}')
    print('[stability] Top 10 por estabilidad:')

    columns = [
        'run_id',
        'algorithm_name',
        'ranking_score',
        'stability_score',
        'stability_intra',
        'stability_inter',
        'stability_gap',
        'n_successful_iterations',
    ]

    print(stability[columns].head(10).to_string(index=False))

    return stability


if __name__ == '__main__':
    pipeline_config = load_pipeline_config()
    experiments_config = load_experiments_config()

    run_stability_analysis(
        pipeline_config=pipeline_config,
        experiments_config=experiments_config,
    )


