# NHTSA ODI Complaint Analytics

Professional-grade data science workspace for analyzing NHTSA ODI consumer complaint data, with a focus on reproducible team workflows, explainable analyses, and future ML/NLP modeling for vehicle safety signal detection.

This repository is designed for a DSBA 6156 (Machine Learning) group project. The current setup emphasizes:

- low-friction onboarding for teammates
- consistent project structure
- easy local extraction and preprocessing of ODI complaint data
- reproducible scripts and shared conventions

The planned analysis scope includes complaint severity prioritization, component-level defect pattern analysis, emerging signal detection, and later time-aware modeling and NLP.

## Project Overview

This project works with NHTSA ODI complaint datasets (complaints first, optional recall joins later). The immediate goal is to standardize repo setup and create a reliable local pipeline that:

1. reads complaint zip files from `data/raw/`
2. extracts tabular source files into `data/extracted/`
3. performs lightweight preprocessing for pandas use
4. writes processed parquet (preferred) or CSV outputs to `data/processed/`
5. writes ingest manifests/summaries to `data/outputs/`

Important data workflow rule:

- raw zip files are treated as immutable source artifacts
- extracted and processed outputs are local workflow artifacts (not intended for Git commit by default)

## Repository Structure

High-level layout:

```text
NHTSA-ODI-COMPLAINT-ANALYTICS/
|-- README.md
|-- requirements.txt
|-- pyproject.toml
|-- .gitignore
|-- .gitattributes
|-- .vscode/
|   |-- extensions.json
|   `-- settings.json
|-- docs/
|   |-- CMPL.txt
|   |-- RCL.txt
|   |-- screenshots/
|   `-- ...
|-- data/
|   |-- raw/         # committed source zips (complaints, optional recalls)
|   |-- extracted/   # local extracted txt/csv files (ignored)
|   |-- processed/   # local parquet/csv outputs (ignored)
|   `-- outputs/     # local run summaries/manifests
|-- scripts/
|   |-- setup_env_windows.ps1
|   |-- setup_env_mac_linux.sh
|   |-- run_pipeline_windows.ps1
|   |-- run_pipeline_mac_linux.sh
|   `-- verify_install.py
|-- notebooks/
|   `-- EDA.ipynb
`-- src/
    |-- __init__.py
    |-- config/
    |   |-- paths.py
    |   |-- constants.py
    |   `-- settings.py
    |-- data/
    |   |-- ingest_odi.py
    |   |-- ingest_recalls.py
    |   |-- schema_checks.py
    |   `-- io_utils.py
    |-- preprocessing/
    |-- features/
    |-- modeling/
    |-- evaluation/
    |-- nlp/
    |-- signals/
    |-- integration/
    `-- reporting/
```

## Repository File/Folder Guide

This section is intentionally detailed for people who may be unfamiliar with Python/data science repos.

### Root files (top level)

`README.md`

- Main onboarding guide for the team
- Use this first when setting up your environment or learning the workflow
- Keep this updated when the team changes setup steps or conventions

`requirements.txt`

- Python packages the project needs (pandas, numpy, scikit-learn, etc)
- Used by the setup scripts to install dependencies into `.venv`
- Team lead may update this file when adding new libraries

`pyproject.toml`

- Tool configuration file (currently used for `ruff` linting rules)
- Most teammates can ignore this unless working on code style or linting config

`.gitignore`

- Tells Git which files should not be committed
- Prevents accidental commits of local outputs and virtual environments

`.gitattributes`

- Helps normalize line endings across Windows/macOS/Linux and marks binary files
- Reduces noisy diffs and cross-platform Git issues

### `.vscode/` (team editor defaults)

`.vscode/extensions.json`

- Recommends shared VS Code extensions for the team
- Most teammates will only use this indirectly by accepting recommendations

`.vscode/settings.json`

- Workspace-level VS Code settings for consistent behavior (format on save, search excludes, Python defaults)
- Windows interpreter path is preconfigured for `.venv\\Scripts\\python.exe`
- macOS/Linux teammates should select `.venv/bin/python` manually

### `.venv/` (local Python environment)

`.venv/`

- Local virtual environment created by the setup scripts
- Contains installed Python packages for this project only
- Not committed to Git
- Most teammates should never edit files inside `.venv/`

### `docs/` (reference docs and project support files)

`docs/CMPL.txt`

- Authoritative complaints dataset schema/data dictionary from NHTSA ODI
- Use this as the source of truth when validating complaint columns or cleaning rules

`docs/RCL.txt`

- Authoritative recalls dataset schema/data dictionary
- Use this later when building recall ingestion and complaint-recall joins

### `data/` (data lifecycle folders)

`data/raw/`

- Immutable source zip files
- Store complaint zips here (and optional recall zips later)
- Do not manually edit or overwrite these files

`data/extracted/`

- Local extracted files produced from raw zips (written directly into this folder)
- Generated by the pipeline and safe to delete/rebuild
- Ignored by Git

`data/processed/`

- Local pandas-friendly outputs (parquet preferred, CSV fallback)
- These are the files your EDA/modeling work will usually read
- Ignored by Git

`data/outputs/`

- Run manifests, summaries, and diagnostics
- Useful for verifying what a pipeline run produced
- Ignored by Git

### `scripts/` (quick setup commands)

`scripts/setup_env_windows.ps1`

- Windows setup automation
- Tries to install Python 3.13 via `winget`, creates `.venv`, installs requirements, runs verification

`scripts/setup_env_mac_linux.sh`

- macOS/Linux setup automation
- Tries `brew`/`apt-get`/`dnf` for Python 3.13, creates `.venv`, installs requirements, runs verification

`scripts/verify_install.py`

- Setup diagnostic script
- Checks Python version, imports, required folders, raw zip presence, and write access

`scripts/run_pipeline_windows.ps1`

- Windows pipeline runner
- Runs verification and then the complaint extraction/preprocessing pipeline

`scripts/run_pipeline_mac_linux.sh`

- macOS/Linux pipeline runner
- Runs verification and then the complaint extraction/preprocessing pipeline

### `notebooks/` (interactive, cell-by-cell exploration)

`notebooks/EDA.ipynb`

- Starter Jupyter notebook for manual EDA in the VS Code editor (good for people who prefer Colab-style workflows)
- Loads processed files from `data/processed/` into pandas DataFrames
- Creates a default DataFrame named `df` (prefers the combined complaints dataset)
- Safe place to explore columns, filters, groupbys, and charts without using the command line

### `src/` ("source" folder, contains main Python files grouped by objective)

`src/__init__.py` and `src/<package>/__init__.py`

- These files mark folders as Python packages so imports work cleanly
  - i.e. the Python files in the folder can be imported the same way you would a library like Pandas with the functions in the file acting like the modules
- Example import: `from src.data.ingest_odi import main`
- They can stay empty and shouldn't need to be edited

`src/config/`

- Shared project configuration and path constants
- `paths.py`: central filesystem paths (repo root, data folders, outputs)
- `constants.py`: project constants and common field-name hints
- `settings.py`: runtime options controlled by environment variables

`src/data/`

- Ingestion and data-validation utilities
- `ingest_odi.py`: current complaint extraction + preprocessing workflow
- `ingest_recalls.py`: starter placeholder for recall extraction
- `schema_checks.py`: doc-driven schema validation (parses `docs/CMPL.txt` and `docs/RCL.txt` to validate columns, types, lengths, dates, and coded values)
- `io_utils.py`: reusable zip/file IO and preprocessing helpers

`src/preprocessing/`

- Cleaning and transformation logic

`src/features/`

- Feature engineering code for ML/NLP/time-based models

`src/modeling/`

- Model training logic (baselines first, then stronger models)

`src/evaluation/`

- Metrics, diagnostics, subgroup checks, and error analysis

`src/nlp/`

- Text/narrative preprocessing and NLP modeling utilities

`src/signals/`

- Early-warning signal, trend, and anomaly detection logic

`src/integration/`

- Cross-dataset joins and integration logic (for example complaints + recalls)

`src/reporting/`

- Reproducible tables/figures/report outputs for presentations and writeups

### Quick rule of thumb for beginners

If you are unsure where to start:

1. Read `README.md`
2. Run the setup script in `scripts/`
3. Run the pipeline script in `scripts/`
4. Inspect outputs in `data/processed/`
5. Explore data in `notebooks/EDA.ipynb`
6. Only then start editing code in `src/`

## Setup Tutorial

This section is intended for teammates who may be new to Python environments, VS Code, or Git.

### 1) Clone the repo in VS Code

Option A (recommended):

1. Open VS Code
2. Press `Ctrl+Shift+P` (Windows) or `Cmd+Shift+P` (macOS)
3. Run `Git: Clone`
4. Paste the repo URL
5. Choose a local folder, should make it where you intend to store repositories
6. Click `Open` when VS Code asks to open the cloned repository

Option B (terminal first, then open in VS Code):

```powershell
git clone <YOUR-REPO-URL>
cd NHTSA-ODI-COMPLAINT-ANALYTICS
code .
```

### 2) Accept the workspace extension recommendations

When VS Code opens the repo, it should detect `.vscode/extensions.json` and prompt for recommended extensions.

Recommended team extensions include Python, Pylance, Rainbow CSV, Ruff, and Data Wrangler.

If you do not see a prompt:

1. Open the Extensions panel
2. Search for `@recommended`
3. Install the Workspace Recommendations

### 3) Run the environment setup script

The setup scripts try to automate Python detection/installation, virtual environment creation, dependency install, and setup verification. Use the terminal pane (defaults to bottom of window) to run the scripts. If there is no terminal window, select Terminal -> New Terminal from the toolbar in the top-left corner.

#### Windows (PowerShell)

Run in the terminal:

```powershell
.\scripts\setup_env_windows.ps1
```

If you get an error because script execution is blocked in PowerShell, run this first:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

#### macOS / Linux

Make scripts executable, then run setup:

```bash
chmod +x scripts/setup_env_mac_linux.sh scripts/run_pipeline_mac_linux.sh
./scripts/setup_env_mac_linux.sh
```

### 4) If setup script cannot install Python automatically (fallback)

The setup scripts check for Python `3.13.12` and will attempt automatic install using common package managers:

- Windows: `winget` (`Python.Python.3.13`)
- macOS: `brew install python@3.13`
- Linux: `apt-get` or `dnf` (if available)

If that fails, install Python manually:

1. Install Python `3.13.12` from sources below
2. Re-open VS Code
3. Re-run the setup script

Manual install sources:

- Windows: https://www.python.org/ftp/python/3.13.12/python-3.13.12-amd64.exe
- macOS: https://www.python.org/ftp/python/3.13.12/python-3.13.12-macos11.pkg

### 5) Run setup verification explicitly (optional)

The setup scripts already run verification, but you can rerun it manually anytime:

#### Windows

```powershell
.\.venv\Scripts\python.exe scripts\verify_install.py
```

#### macOS / Linux

```bash
.venv/bin/python scripts/verify_install.py
```

What `scripts/verify_install.py` checks:

- Python version (warns if not exactly `3.13.12`)
- core imports (`pandas`, `numpy`, `sklearn`, `matplotlib`, `pyarrow`)
- key project folders
- presence of zip files in `data/raw/`
- write access to `data/outputs/`

### 6) Activate the virtual environment

The setup scripts install dependencies using the venv Python directly, but if you want to run commands interactively (Jupyter, scripts, notebooks), activate the venv in your terminal. This is something you will need to do each time you fully close VS Code/your terminal if you want to run commands from the terminal.

#### Windows (PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
source .venv/bin/activate
```

### 7) VS Code interpreter note (important for macOS/Linux users)

The workspace setting in `.vscode/settings.json` sets the default interpreter path to the Windows venv path:

- `${workspaceFolder}\\.venv\\Scripts\\python.exe`

If you are on macOS/Linux, select your interpreter manually in VS Code:

1. `Ctrl/Cmd + Shift + P`
2. `Python: Select Interpreter`
3. Choose `.venv/bin/python`

## Data Setup Instructions

Make sure raw NHTSA ODI zip files are in `data/raw/`.

Authoritative schema/data dictionary references used by the project:

- complaints: `docs/CMPL.txt`
- recalls: `docs/RCL.txt`

Use these docs when proposing schema changes, writing cleaning rules, or debugging missing/renamed columns.

Preferred complaint filenames for team consistency:

- `data/raw/COMPLAINTS_RECEIVED_2020-2024.zip`
- `data/raw/COMPLAINTS_RECEIVED_2025-2026.zip`

The ingestion script expects these standard complaint zip filenames exactly.

Optional recall files may also be stored in `data/raw/` for later join work. A starter recall extraction script placeholder exists in `src/data/ingest_recalls.py`.

Important local-processing design choices:

- raw zip files are committed and preserved as source data
- extraction happens locally into `data/extracted/`
- processed outputs are written locally to `data/processed/`
- run manifests/summaries are written locally to `data/outputs/`
- these derived folders are ignored by Git to avoid oversized commits and GitHub file-size issues

## How To Run The Project (Standard Workflow)

### Standard script order

1. Run setup script (`setup_env_*`)
2. Confirm verification passes (`verify_install.py` runs automatically)
3. Run the pipeline script (`run_pipeline_*`)
4. Inspect outputs in `data/processed/` and manifests in `data/outputs/`

### Run the pipeline (Windows)

```powershell
.\scripts\run_pipeline_windows.ps1
```

Optional flags (changes output from parquet to csv, overwrites existing extracted files, prevents file combining):

```powershell
.\scripts\run_pipeline_windows.ps1 -OutputFormat csv
.\scripts\run_pipeline_windows.ps1 -OverwriteExtracted
.\scripts\run_pipeline_windows.ps1 -NoCombine
```

### Run the pipeline (macOS / Linux)

```bash
./scripts/run_pipeline_mac_linux.sh
```

Optional flags (changes output from parquet to csv, overwrites existing extracted files, prevents file combining):

```bash
./scripts/run_pipeline_mac_linux.sh --output-format csv
./scripts/run_pipeline_mac_linux.sh --overwrite-extracted
./scripts/run_pipeline_mac_linux.sh --no-combine
```

### What the current starter pipeline does

`src/data/ingest_odi.py` currently:

- discovers complaint zips in `data/raw/`
- extracts files directly to `data/extracted/`
- reads `.txt/.csv/.tsv` files into pandas
- assigns official complaint schema column names from `docs/CMPL.txt`
- applies minor preprocessing (trim strings, parse some date-like fields, coerce common model year columns)
- writes processed parquet (preferred) or CSV outputs to `data/processed/`
- optionally builds a combined complaint dataset
- writes run manifests and simple summaries to `data/outputs/`

## Git Basics Overview

### VS Code Source Control

Most teammates can do daily Git work using the VS Code Source Control panel:

1. Click the Source Control icon on the left sidebar (3rd from the top)
2. Review changed files
3. Stage files by clicking `+` (shows by hovering over file name, or all at once by hovering over `Changes`)
4. Enter a commit message
5. Click `Commit`
6. Click `Sync Changes` (or `Push`) to upload your branch
7. Click `Pull` before starting work or before pushing if others may have updated the branch

Good habit before you start coding:

- open Source Control
- click `Pull` (or `Sync Changes`)
- confirm no incoming changes remain

### Command Line Git commands (examples)

Check your state:

```bash
git status
git status -sb
```

Pull latest changes for your current branch:

```bash
git pull
```

Stage + commit + push:

```bash
git add .
git commit -m "Add initial ODI preprocessing notebook scaffold"
git push
```

### Branch workflow

Do not do project work directly on `main`.

Use this workflow instead:

1. Update `main`
2. Create a branch from `main`
3. Do your work on that branch
4. Commit your changes
5. Pull the latest `main` into your branch before finishing
6. Push your branch to GitHub

### Create a new branch (Command Line version)

Start from an up-to-date `main` branch:

```bash
git switch main
git pull origin main
git switch -c feature/short-description
```

Example:

```bash
git switch main
git pull origin main
git switch -c feature/schema-quality-checks
```

### Create a new branch (VS Code)

1. Make sure you are on `main` (check the branch name in the bottom-left of VS Code)
2. Pull latest changes on `main` (`Source Control` -> `...` menu -> `Pull`, or click `Sync Changes`)
3. Click the branch name in the bottom-left status bar
4. Choose `Create new branch...`
5. Name it something like `feature/...`, `eda/...`, or `docs/...`

### Work on your branch and commit changes

Example Command Line flow while working:

```bash
git status
git add .
git commit -m "Add complaint schema validation summaries"
```

You can make multiple commits on the same branch while you work.

### Proper way to pull `main` into your branch (before finishing or opening a PR)

This helps reduce merge conflicts and makes sure your branch includes recent team changes.

Important first step:

- Commit your work (or stash it) before updating from `main`

Recommended Command Line workflow (merge `main` into your branch):

```bash
git switch main
git pull origin main
git switch feature/short-description
git merge main
```

Alternative (equivalent result, fewer branch switches):

```bash
git fetch origin
git switch feature/short-description
git merge origin/main
```

If there are merge conflicts:

1. Open the conflicted files in VS Code
2. Use the conflict resolution buttons (`Accept Current`, `Accept Incoming`, `Accept Both`) carefully
3. Run `git status` to confirm what is still unresolved
4. Stage resolved files
5. Commit the merge (if Git does not auto-create the commit for you)

Example conflict-resolution finish:

```bash
git add .
git commit -m "Merge main into feature/short-description"
```

VS Code way to bring `main` into your branch:

1. Commit or stash your current changes on your branch
2. Switch to `main`
3. Pull latest `main`
4. Switch back to your branch
5. Open Command Palette and run `Git: Merge Branch...`
6. Select `main`
7. Resolve conflicts if prompted

### Push your branch when you are finished (or ready for review)

First push of a new branch (sets the upstream tracking branch):

```bash
git push -u origin feature/short-description
```

Later pushes on the same branch:

```bash
git push
```

VS Code push flow:

1. Commit your changes
2. Click `Publish Branch` (first push) or `Sync Changes` / `Push` (later pushes)
3. Confirm the branch name matches your feature branch, not `main`

### End-of-work checklist for teammates

Before you say "I’m done", do this:

1. `git status` is clean (or only shows files you intentionally left uncommitted)
2. You pulled/merged latest `main` into your branch
3. You resolved any conflicts
4. You pushed your branch to GitHub
5. You shared the branch name with the team (or opened a PR, if using PRs)

### How to tell if the branch was updated and you need to pull

In VS Code:

- the status bar / Source Control view may show incoming changes
- the `Sync`/cloud arrows indicator can show incoming commits
- after clicking `Refresh` or `Fetch`, if you see incoming changes, pull first

In Command Line (more explicit):

```bash
git fetch
git status -sb
git branch -vv
```

If your branch line shows it is behind `origin/<branch>` (for example `behind 2`), you need to pull.

You can also inspect incoming commits directly:

```bash
git log --oneline HEAD..origin/<your-branch-name>
```

If this prints commits, your local branch is behind the remote branch.

## Team Workflow Rules

Use these rules to reduce merge conflicts and lost work in a class group setting:

1. Pull before you start work each session
2. Create a branch for your task (do not do all work on the shared branch)
3. Keep commits small and focused (one topic per commit when possible)
4. Avoid editing the same file as someone else at the same time
5. Do not edit or rewrite committed raw zip files in `data/raw/`
6. Do not commit `data/extracted/`, `data/processed/`, or `data/outputs/` artifacts
7. If you change shared schemas or feature logic, document the change in your PR/commit message
8. Pull again before pushing if you were working for a while
