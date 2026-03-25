from pathlib import Path

from ymz304_project.cli import main
from ymz304_project.experiment import run_experiments


def test_run_experiments_creates_summary_outputs_for_all_frameworks(tmp_path: Path) -> None:
    result = run_experiments(
        output_dir=tmp_path,
        numpy_epochs=20,
        sklearn_epochs=20,
        torch_epochs=20,
    )

    # 4 model × 3 framework (standard) + 1 model × 3 framework (minmax) = 15
    assert len(result["results"]) == 15
    assert {item["framework"] for item in result["results"]} == {"numpy", "sklearn", "torch"}
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "model_comparison.csv").exists()
    assert any(path.name.startswith("confusion_matrix_") for path in tmp_path.iterdir())
    assert any(path.name.startswith("learning_curve_") for path in tmp_path.iterdir())


def test_cli_main_runs_pipeline_and_returns_success(tmp_path: Path) -> None:
    exit_code = main(
        [
            "--output-dir",
            str(tmp_path),
            "--numpy-epochs",
            "5",
            "--sklearn-epochs",
            "5",
            "--torch-epochs",
            "5",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "summary.json").exists()
