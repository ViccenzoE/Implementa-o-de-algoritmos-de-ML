from pathlib import Path

from src.train import run


def test_pipeline_returns_all_models_and_artifacts(tmp_path: Path) -> None:
    results = run(output_dir=tmp_path)

    assert len(results) == 6
    assert {"accuracy", "precision", "recall", "f1_score"} <= set(results.columns)
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "metrics_comparison.png").exists()
    assert (tmp_path / "confusion_matrices.png").exists()
