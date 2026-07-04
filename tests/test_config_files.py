
import yaml


def load_yaml(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def test_pipeline_config_is_valid_yaml():
    config = load_yaml('config/pipeline.yaml')

    assert config['proyecto']['nombre'] == 'chili-clustering-benchmark'
    assert config['dataset']['periodo_inicio'] == '2011-12-01'
    assert config['dataset']['periodo_fin'] == '2025-11-30'
    assert len(config['embalses']['sufijos']) == 7
    assert len(config['features']['base']) == 5


def test_experiments_config_is_valid_yaml():
    config = load_yaml('config/experiments.yaml')

    assert config['experimento']['herramienta_tracking'] == 'mlflow'
    assert config['mlflow']['experiment_name'] == 'chili_clustering_benchmark'
    assert len(config['algoritmos']) >= 5
    assert config['metricas']['estabilidad']['activo'] is True


def test_all_main_algorithms_have_grid():
    config = load_yaml('config/experiments.yaml')

    for name, spec in config['algoritmos'].items():
        assert 'activo' in spec
        assert 'familia' in spec
        assert 'justificacion' in spec
        assert 'grid' in spec
        assert spec['grid'], f'El algoritmo {name} no tiene grilla.'
