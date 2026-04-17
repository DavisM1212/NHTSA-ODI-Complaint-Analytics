#!/usr/bin/env bash
set -euo pipefail

TASK_TYPE="${TASK_TYPE:-CPU}"
DEVICES="${DEVICES:-0}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-parquet}"
SKIP_INGEST="${SKIP_INGEST:-0}"
SKIP_VISUALS="${SKIP_VISUALS:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  PYTHON_BIN="python"
fi

run_module() {
  local module_name="$1"
  shift
  printf '\n==> Running %s\n' "$module_name"
  "$PYTHON_BIN" -m "$module_name" "$@"
}

printf 'NHTSA ODI Complaint Analytics - Official component pipeline\n'
printf 'Repository root: %s\n' "$REPO_ROOT"
printf 'Python: %s\n' "$PYTHON_BIN"
printf 'Task type: %s\n' "$TASK_TYPE"

printf '\n==> Running install verification\n'
"$PYTHON_BIN" scripts/verify_install.py

if [[ "$SKIP_INGEST" != "1" ]]; then
  run_module src.data.ingest_odi --output-format "$OUTPUT_FORMAT"
fi

run_module src.preprocessing.clean_complaints
run_module src.features.collapse_components
run_module src.features.component_text_sidecar
run_module src.modeling.official.component_single_text_calibrated --task-type "$TASK_TYPE" --devices "$DEVICES"
run_module src.modeling.official.component_multi_routing --task-type "$TASK_TYPE" --devices "$DEVICES"
run_module src.reporting.update_component_readme

if [[ "$SKIP_VISUALS" != "1" ]]; then
  run_module src.reporting.component_visuals
fi

printf '\nOfficial component pipeline completed\n'
printf 'Check data/processed/, data/outputs/, and docs/figures/component_models/\n'
