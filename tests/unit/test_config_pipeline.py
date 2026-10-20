"""Unit tests for config-driven experiment execution and corpus I/O."""

from __future__ import annotations

from pathlib import Path

from experiments.scripts.run_synthetic_safe_ceg import _load_config, run_experiment
from src.training.workload import (
    generate_synthetic_workload,
    load_workload_corpus,
    save_workload_corpus,
)


def test_workload_corpus_round_trip_yaml(tmp_path: Path) -> None:
    workload = generate_synthetic_workload(
        family="path",
        query_count=3,
        relation_count=4,
        seed=5,
    )
    corpus_path = tmp_path / "workload.yaml"

    save_workload_corpus(workload, corpus_path)
    loaded_workload = load_workload_corpus(corpus_path)

    assert loaded_workload == workload


def test_load_config_reads_synthetic_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(
        """
experiment:
  name: test-config
  seed: 7
synthetic_workload:
  train_family: path
  eval_family: cycle
  train_count: 4
  eval_count: 2
  relation_count: 4
  train_corpus_path: null
  eval_corpus_path: null
training:
  epochs: 2
  learning_rate: 0.001
  safety_lambda: 0.5
  safety_tau: 2.0
output:
  summary_path: results/test.json
""".strip()
        + "\n"
    )

    config = _load_config(config_path)

    assert config.name == "test-config"
    assert config.train_family == "path"
    assert config.eval_family == "cycle"
    assert config.output_path == Path("results/test.json")


def test_run_experiment_uses_persisted_corpora(tmp_path: Path) -> None:
    from experiments.scripts.run_synthetic_safe_ceg import SyntheticExperimentConfig

    train_corpus = tmp_path / "train.yaml"
    eval_corpus = tmp_path / "eval.yaml"
    save_workload_corpus(
        generate_synthetic_workload("path", query_count=4, relation_count=4, seed=2),
        train_corpus,
    )
    save_workload_corpus(
        generate_synthetic_workload("cycle", query_count=2, relation_count=4, seed=3),
        eval_corpus,
    )

    payload = run_experiment(
        SyntheticExperimentConfig(
            name="corpus-test",
            seed=2,
            train_family="path",
            eval_family="cycle",
            train_count=4,
            eval_count=2,
            relation_count=4,
            epochs=2,
            learning_rate=1e-3,
            tau=2.0,
            safety_lambda=0.5,
            output_path=tmp_path / "summary.json",
            train_corpus_path=train_corpus,
            eval_corpus_path=eval_corpus,
        )
    )

    assert payload["train_count"] == 4
    assert payload["eval_count"] == 2
    assert "evaluation" in payload
