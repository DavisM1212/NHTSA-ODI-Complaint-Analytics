"""Install local git filters used by this repo"""

import subprocess
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def run_git_config(key, value):
    cmd = ["git", "config", "--local", key, value]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    if (REPO_ROOT / ".venv" / "Scripts" / "python.exe").exists():
        python_cmd = ".venv/Scripts/python.exe"
    elif (REPO_ROOT / ".venv" / "bin" / "python").exists():
        python_cmd = ".venv/bin/python"
    else:
        python_cmd = Path(sys.executable).resolve().as_posix()

    filter_cmd = "scripts/git_notebook_filter.py"

    clean_cmd = f'"{python_cmd}" "{filter_cmd}" --mode clean'
    smudge_cmd = f'"{python_cmd}" "{filter_cmd}" --mode smudge'

    run_git_config("filter.notebookstrip.clean", clean_cmd)
    run_git_config("filter.notebookstrip.smudge", smudge_cmd)
    run_git_config("filter.notebookstrip.required", "true")

    print("Installed local git filter: notebookstrip")
    print("Exploration notebooks configured in .gitattributes will be cleared on commit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
