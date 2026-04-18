$ErrorActionPreference = "Stop"

. .\.venv\Scripts\Activate.ps1
python -m pytest -q tests -p no:cacheprovider
python -m src.run_experiments
