from pathlib import Path
from typing import Any

import yaml


DEFAULT_PIPELINE_CONFIG = Path('config/pipeline.yaml')
DEFAULT_EXPERIMENTS_CONFIG = Path('config/experiments.yaml')


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its content as a dictionary."""
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f'Config file not found: {config_path}')

    with config_path.open('r', encoding='utf-8') as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f'Config file must contain a mapping: {config_path}')

    return data


def validate_required_sections(
    config: dict[str, Any],
    required_sections: list[str],
    config_name: str,
) -> None:
    """Validate that a config dictionary contains required top-level sections."""
    missing = [section for section in required_sections if section not in config]

    if missing:
        missing_text = ', '.join(missing)
        raise ValueError(
            f'Missing required sections in {config_name}: {missing_text}'
        )


def load_pipeline_config(
    path: str | Path = DEFAULT_PIPELINE_CONFIG,
) -> dict[str, Any]:
    """Load and validate the main pipeline configuration."""
    config = load_yaml(path)
    validate_required_sections(
        config=config,
        required_sections=[
            'proyecto',
            'rutas',
            'dataset',
            'embalses',
            'variables',
            'archivo_fuente',
            'features',
        ],
        config_name='pipeline config',
    )
    return config


def load_experiments_config(
    path: str | Path = DEFAULT_EXPERIMENTS_CONFIG,
) -> dict[str, Any]:
    """Load and validate the experiment benchmark configuration."""
    config = load_yaml(path)
    validate_required_sections(
        config=config,
        required_sections=[
            'experimento',
            'mlflow',
            'preprocesamiento_experimental',
            'algoritmos',
            'metricas',
            'restricciones_particion',
            'ranking',
            'salidas',
        ],
        config_name='experiments config',
    )
    return config


def ensure_output_directories(pipeline_config: dict[str, Any]) -> list[Path]:
    """Create output directories declared in the pipeline configuration."""
    rutas = pipeline_config['rutas']

    directories = [
        Path('data/raw'),
        Path('data/processed'),
        Path(rutas['carpeta_tablas']),
        Path(rutas['carpeta_figuras']),
        Path(rutas['carpeta_reportes']),
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    return directories
