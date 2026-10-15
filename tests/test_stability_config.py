from src.config_loader import load_experiments_config


def test_experiments_config_has_stability_section():
    config = load_experiments_config()

    assert 'estabilidad' in config
    assert config['estabilidad']['metodo'] == 'exhaustivo_sin_reemplazo'
    assert config['estabilidad']['max_combinaciones'] == 10000
    assert config['estabilidad']['fraccion_muestra'] == 0.85
