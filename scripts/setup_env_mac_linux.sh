#!/usr/bin/env bash
set -euo pipefail

TARGET_PYTHON_VERSION="${TARGET_PYTHON_VERSION:-3.13.12}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

print_step() {
  printf '\n==> %s\n' "$1"
}

python_version_of() {
  local py_bin="$1"
  "$py_bin" --version 2>&1 | awk '{print $2}'
}

find_python_313() {
  if command -v python3.13 >/dev/null 2>&1; then
    echo "python3.13"
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    local version
    version="$(python_version_of python3 || true)"
    if [[ "$version" == 3.13.* ]]; then
      echo "python3"
      return 0
    fi
  fi

  return 1
}

install_python_313_if_possible() {
  local uname_out
  uname_out="$(uname -s)"

  if [[ "$uname_out" == "Darwin" ]]; then
    if command -v brew >/dev/null 2>&1; then
      print_step "Attempting Python install with Homebrew (python@3.13)"
      brew install python@3.13 || return 1
      return 0
    fi
    return 1
  fi

  if [[ -f /etc/os-release ]]; then
    if command -v apt-get >/dev/null 2>&1; then
      print_step "Attempting Python install with apt-get (python3.13)"
      if command -v sudo >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y python3.13 python3.13-venv || return 1
      else
        apt-get update
        apt-get install -y python3.13 python3.13-venv || return 1
      fi
      return 0
    fi

    if command -v dnf >/dev/null 2>&1; then
      print_step "Attempting Python install with dnf (python3.13)"
      if command -v sudo >/dev/null 2>&1; then
        sudo dnf install -y python3.13 || return 1
      else
        dnf install -y python3.13 || return 1
      fi
      return 0
    fi
  fi

  return 1
}

printf 'NHTSA ODI Complaint Analytics - macOS/Linux environment setup\n'
printf 'Repository root: %s\n' "$REPO_ROOT"
cd "$REPO_ROOT"

print_step "Checking Python"
PYTHON_BIN="$(find_python_313 || true)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  printf 'Python 3.13 was not detected\n'
  if install_python_313_if_possible; then
    PYTHON_BIN="$(find_python_313 || true)"
  fi
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  printf '\nAutomatic install failed or Python 3.13 is still unavailable\n'
  printf 'Manual install steps\n'
  printf '1) Install Python %s (or latest Python 3.13.x) using python.org, pyenv, or your OS package manager\n' "$TARGET_PYTHON_VERSION"
  printf '2) Re-open your terminal\n'
  printf '3) Re-run ./scripts/setup_env_mac_linux.sh\n'
  exit 1
fi

PYTHON_VERSION="$("$PYTHON_BIN" --version 2>&1 | awk '{print $2}')"
printf 'Using interpreter: %s\n' "$PYTHON_BIN"
printf 'Detected version: %s\n' "$PYTHON_VERSION"
if [[ "$PYTHON_VERSION" != "$TARGET_PYTHON_VERSION" ]]; then
  printf 'Warning: recommended version is %s, but continuing with %s\n' "$TARGET_PYTHON_VERSION" "$PYTHON_VERSION"
fi

print_step "Creating virtual environment (.venv) if needed"
if [[ ! -x ".venv/bin/python" ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

print_step "Upgrading pip"
".venv/bin/python" -m pip install --upgrade pip

print_step "Installing requirements.txt"
".venv/bin/python" -m pip install -r requirements.txt

print_step "Running install verification"
set +e
".venv/bin/python" scripts/verify_install.py
VERIFY_EXIT_CODE=$?
set -e

printf '\n'
if [[ $VERIFY_EXIT_CODE -eq 0 ]]; then
  printf 'Setup completed successfully\n'
  printf 'Next step: ./scripts/run_pipeline_mac_linux.sh\n'
else
  printf 'Setup completed with verification failures\n'
  printf 'Review the messages above, fix the issues, then rerun this script\n'
fi

printf 'Tip: Activate the venv in a new shell with source .venv/bin/activate\n'
exit $VERIFY_EXIT_CODE
