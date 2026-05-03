#!/usr/bin/env bash
set -euo pipefail

OUTPUT_FORMAT="${OUTPUT_FORMAT:-parquet}"
SKIP_INGEST="${SKIP_INGEST:-0}"
SKIP_CACHE_REBUILD="${SKIP_CACHE_REBUILD:-0}"
RANDOM_SEED="${RANDOM_SEED:-42}"
PUBLISH_STATUS="${PUBLISH_STATUS:-official}"

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

printf 'NHTSA ODI Complaint Analytics - Official NLP early-warning pipeline\n'
printf 'Repository root: %s\n' "$REPO_ROOT"
printf 'Python: %s\n' "$PYTHON_BIN"
printf 'Random seed: %s\n' "$RANDOM_SEED"

printf '\n==> Running install verification\n'
"$PYTHON_BIN" scripts/verify_install.py

if [[ "$SKIP_INGEST" != "1" ]]; then
  run_module src.data.ingest_odi --output-format "$OUTPUT_FORMAT"
fi

run_module src.preprocessing.clean_complaints

NLP_ARGS=(--random-seed "$RANDOM_SEED" --publish-status "$PUBLISH_STATUS")
if [[ "$SKIP_CACHE_REBUILD" == "1" ]]; then
  NLP_ARGS+=(--skip-cache-rebuild)
fi

run_module src.modeling.nlp_early_warning_system "${NLP_ARGS[@]}"
run_module src.reporting.watchlist_visuals

printf '\nOfficial NLP early-warning pipeline completed\n'
printf 'Check data/processed/odi_nlp_prepped.parquet, data/outputs/, and docs/figures/nlp_early_warning/\n'
