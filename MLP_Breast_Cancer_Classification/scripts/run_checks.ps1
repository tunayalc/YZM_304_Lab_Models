$ErrorActionPreference = "Stop"

python -m pytest
python -m ruff check .
python .\run_project.py --output-dir artifacts\check-run
