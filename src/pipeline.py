from __future__ import annotations

import argparse

from src.algorithm_analysis import run_algorithm_analysis
from src.benchmark import run_benchmark
from src.config_loader import load_experiments_config, load_pipeline_config
from src.data_summary import run_data_summary
from src.figures import generate_figures
from src.final_ranking import run_final_ranking
from src.interpretation import run_interpretation
from src.partition_equivalence import run_partition_equivalence
from src.ranking import run_ranking
from src.stability import run_stability_analysis


PIPELINE_STEPS = [
    'benchmark',
    'data_summary',
    'ranking',
    'partition_equivalence',
    'stability',
    'final_ranking',
    'algorithm_analysis',
    'interpretation',
    'figures',
]


def print_step_start(step_name: str) -> None:
    """Print a consistent pipeline step header."""
    print('')
    print(f'[pipeline] Iniciando paso: {step_name}')


def print_step_done(step_name: str) -> None:
    """Print a consistent pipeline step completion message."""
    print(f'[pipeline] Paso completado: {step_name}')


def run_pipeline(
    log_to_mlflow: bool = True,
    max_stability_runs: int | None = None,
) -> list[str]:
    """Run the full reproducible clustering benchmark pipeline."""
    pipeline_config = load_pipeline_config()
    experiments_config = load_experiments_config()

    completed_steps: list[str] = []

    print('[pipeline] Ejecución reproducible iniciada')
    print(f'[pipeline] MLflow habilitado: {log_to_mlflow}')

    print_step_start('benchmark')
    run_benchmark(
        pipeline_config=pipeline_config,
        experiments_config=experiments_config,
        log_to_mlflow=log_to_mlflow,
    )
    completed_steps.append('benchmark')
    print_step_done('benchmark')

    print_step_start('data_summary')
    run_data_summary(pipeline_config)
    completed_steps.append('data_summary')
    print_step_done('data_summary')

    print_step_start('ranking')
    run_ranking(experiments_config)
    completed_steps.append('ranking')
    print_step_done('ranking')

    print_step_start('partition_equivalence')
    run_partition_equivalence(experiments_config)
    completed_steps.append('partition_equivalence')
    print_step_done('partition_equivalence')

    print_step_start('stability')
    run_stability_analysis(
        pipeline_config=pipeline_config,
        experiments_config=experiments_config,
        max_runs=max_stability_runs,
    )
    completed_steps.append('stability')
    print_step_done('stability')

    print_step_start('final_ranking')
    run_final_ranking(experiments_config)
    completed_steps.append('final_ranking')
    print_step_done('final_ranking')

    print_step_start('algorithm_analysis')
    run_algorithm_analysis(experiments_config)
    completed_steps.append('algorithm_analysis')
    print_step_done('algorithm_analysis')

    print_step_start('interpretation')
    run_interpretation(
        pipeline_config=pipeline_config,
        experiments_config=experiments_config,
    )
    completed_steps.append('interpretation')
    print_step_done('interpretation')

    print_step_start('figures')
    generate_figures(experiments_config)
    completed_steps.append('figures')
    print_step_done('figures')

    print('')
    print('[pipeline] Ejecución reproducible completada')
    print('[pipeline] Pasos ejecutados:')
    for step in completed_steps:
        print(f'[pipeline] - {step}')

    return completed_steps


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the pipeline runner."""
    parser = argparse.ArgumentParser(
        description='Run the reproducible Chili clustering benchmark pipeline.'
    )
    parser.add_argument(
        '--no-mlflow',
        action='store_true',
        help='Run the benchmark without logging runs to MLflow.',
    )
    parser.add_argument(
        '--max-stability-runs',
        type=int,
        default=None,
        help='Optional limit for stability runs, useful for smoke tests.',
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> list[str]:
    """Command-line entrypoint."""
    args = parse_args(argv)

    return run_pipeline(
        log_to_mlflow=not args.no_mlflow,
        max_stability_runs=args.max_stability_runs,
    )


if __name__ == '__main__':
    main()
