from src.config_loader import load_experiments_config


def test_ranking_weights_include_stability_weight():
    config = load_experiments_config()

    weights = config['ranking']['pesos']

    assert 'estabilidad' in weights
    assert weights['estabilidad'] == 0.25
