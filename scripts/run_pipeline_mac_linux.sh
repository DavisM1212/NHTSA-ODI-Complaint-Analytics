#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  PYTHON_BIN="python"
fi

printf 'NHTSA ODI Complaint Analytics - macOS/Linux pipeline runner\n'
printf 'Repository root: %s\n' "$REPO_ROOT"
printf 'Python: %s\n' "$PYTHON_BIN"

printf '\n==> Running install verification\n'
"$PYTHON_BIN" scripts/verify_install.py

printf '\n==> Running ODI complaint extraction + preprocessing\n'
"$PYTHON_BIN" -m src.data.ingest_odi "$@"

printf '\nPipeline completed\n'
printf 'Check data/extracted/, data/processed/, and data/outputs/\n'
