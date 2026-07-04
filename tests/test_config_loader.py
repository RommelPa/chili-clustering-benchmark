from pathlib import Path

import pytest

from src.config_loader import (
    ensure_output_directories,
    load_experiments_config,
    load_pipeline_config,
    load_yaml,
    validate_required_sections,
)


def test_load_pipeline_config_has_project_name():
    config = load_pipeline_config()

    assert config['proyecto']['nombre'] == 'chili-clustering-benchmark'


def test_load_experiments_config_has_algorithms():
    config = load_experiments_config()

    assert len(config['algoritmos']) >= 5
    assert 'kmeans' in config['algoritmos']
    assert 'dbscan' in config['algoritmos']


def test_load_yaml_raises_for_missing_file():
    with pytest.raises(FileNotFoundError):
        load_yaml('config/no_existe.yaml')


def test_validate_required_sections_raises_for_missing_section():
    with pytest.raises(ValueError, match='Missing required sections'):
        validate_required_sections(
            config={'proyecto': {}},
            required_sections=['proyecto', 'rutas'],
            config_name='test config',
        )


def test_ensure_output_directories_returns_existing_paths():
    config = load_pipeline_config()
    directories = ensure_output_directories(config)

    assert directories
    assert all(isinstance(path, Path) for path in directories)
    assert all(path.exists() for path in directories)
