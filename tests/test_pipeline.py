from src import pipeline


def test_parse_args_supports_no_mlflow_and_stability_limit():
    args = pipeline.parse_args(
        [
            '--no-mlflow',
            '--max-stability-runs',
            '5',
        ]
    )

    assert args.no_mlflow is True
    assert args.max_stability_runs == 5


def test_run_pipeline_calls_steps_in_expected_order(monkeypatch):
    calls: list[str] = []
    pipeline_config = {'config': 'pipeline'}
    experiments_config = {'config': 'experiments'}

    monkeypatch.setattr(
        pipeline,
        'load_pipeline_config',
        lambda: pipeline_config,
    )
    monkeypatch.setattr(
        pipeline,
        'load_experiments_config',
        lambda: experiments_config,
    )

    def fake_run_benchmark(
        pipeline_config,
        experiments_config,
        log_to_mlflow,
    ):
        assert pipeline_config == {'config': 'pipeline'}
        assert experiments_config == {'config': 'experiments'}
        assert log_to_mlflow is False
        calls.append('benchmark')

    def fake_run_data_summary(config):
        assert config == {'config': 'pipeline'}
        calls.append('data_summary')

    def fake_run_ranking(config):
        assert config == {'config': 'experiments'}
        calls.append('ranking')

    def fake_run_partition_equivalence(config):
        assert config == {'config': 'experiments'}
        calls.append('partition_equivalence')

    def fake_run_stability_analysis(
        pipeline_config,
        experiments_config,
        max_runs,
    ):
        assert pipeline_config == {'config': 'pipeline'}
        assert experiments_config == {'config': 'experiments'}
        assert max_runs == 3
        calls.append('stability')

    def fake_run_final_ranking(config):
        assert config == {'config': 'experiments'}
        calls.append('final_ranking')

    def fake_run_algorithm_analysis(config):
        assert config == {'config': 'experiments'}
        calls.append('algorithm_analysis')

    def fake_run_interpretation(pipeline_config, experiments_config):
        assert pipeline_config == {'config': 'pipeline'}
        assert experiments_config == {'config': 'experiments'}
        calls.append('interpretation')

    def fake_generate_figures(config):
        assert config == {'config': 'experiments'}
        calls.append('figures')

    monkeypatch.setattr(pipeline, 'run_benchmark', fake_run_benchmark)
    monkeypatch.setattr(pipeline, 'run_data_summary', fake_run_data_summary)
    monkeypatch.setattr(pipeline, 'run_ranking', fake_run_ranking)
    monkeypatch.setattr(
        pipeline,
        'run_partition_equivalence',
        fake_run_partition_equivalence,
    )
    monkeypatch.setattr(
        pipeline,
        'run_stability_analysis',
        fake_run_stability_analysis,
    )
    monkeypatch.setattr(pipeline, 'run_final_ranking', fake_run_final_ranking)
    monkeypatch.setattr(
        pipeline,
        'run_algorithm_analysis',
        fake_run_algorithm_analysis,
    )
    monkeypatch.setattr(pipeline, 'run_interpretation', fake_run_interpretation)
    monkeypatch.setattr(pipeline, 'generate_figures', fake_generate_figures)

    completed_steps = pipeline.run_pipeline(
        log_to_mlflow=False,
        max_stability_runs=3,
    )

    assert calls == pipeline.PIPELINE_STEPS
    assert completed_steps == pipeline.PIPELINE_STEPS
