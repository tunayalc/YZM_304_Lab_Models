from __future__ import annotations

import argparse
from pathlib import Path

from ymz304_project.experiment import run_experiments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YZM304 derin ogrenme deneylerini calistirir.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Cikti klasoru")
    parser.add_argument("--numpy-epochs", type=int, default=250, help="NumPy model epoch sayisi")
    parser.add_argument(
        "--sklearn-epochs",
        type=int,
        default=250,
        help="Scikit-learn model epoch sayisi",
    )
    parser.add_argument("--torch-epochs", type=int, default=250, help="PyTorch model epoch sayisi")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    run_experiments(
        output_dir=args.output_dir,
        numpy_epochs=args.numpy_epochs,
        sklearn_epochs=args.sklearn_epochs,
        torch_epochs=args.torch_epochs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
