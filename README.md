# NHTSA ODI Complaint Analytics

Professional-grade data science workspace for analyzing National Highway Traffic Safety Administration (NHTSA) Office of Defects Investigation (ODI) consumer complaint data, with a focus on reproducible workflows, explainable analyses, and leakage-aware modeling for vehicle safety signal detection.

This repository is designed for a DSBA 6156 (Machine Learning) group project. The current setup emphasizes:

- low-friction onboarding for teammates
- consistent project structure
- easy local extraction, cleaning, and modeling of ODI complaint data
- reproducible scripts and shared conventions

The current workflow already covers complaint ingestion, EDA, audited cleaning, single-label and multi-label component target construction, structured component benchmarks, and a promoted complaint-narrative single-label component model. The next major steps are complaint severity ranking and NLP-driven early warning beyond component classification.

## Project Overview

This project works with NHTSA ODI complaint datasets (complaints first, optional recall joins later). The current pipeline is designed to:

1. read complaint zip files from `data/raw/`
2. extract tabular source files into `data/extracted/`
3. build processed complaint tables in `data/processed/`
4. apply conservative, schema-aware cleaning and issue flagging
5. build task-specific modeling tables for severity and component work
6. benchmark component models with time-aware validation

Important data workflow rule:

- raw zip files are treated as immutable source artifacts
- extracted and processed outputs are local workflow artifacts (not intended for Git commit by default)

## Current Status

The repo is now beyond the initial scaffold stage. Current completed work includes:

- `notebooks/EDA.ipynb`: structural audit of the complaint data
- `notebooks/Cleaning.ipynb`: reviewed cleaning decisions tied to `docs/CMPL.txt`
- `src/preprocessing/clean_complaints.py`: reproducible shared cleaning pipeline
- `src/features/collapse_components.py`: single-label and multi-label component case table builder
- `notebooks/Component_Modeling.ipynb`: exploratory modeling workflow for the original single-label benchmark
- `src/modeling/tune_component_catboost.py`: canonical single-label feature selection + focused CatBoost model selection
- `src/modeling/component_catboost.py`: final single-label benchmark with baselines and untouched 2026 holdout scoring
- `src/modeling/component_multilabel.py`: multi-label routing benchmark on the full kept-case problem
- `src/features/component_text_sidecar.py`: shared component narrative sidecar for leakage-aware text modeling
- `src/modeling/component_text_wave2.py`: Wave 2 text-feature experiment runner for component modeling
- `src/modeling/component_text_wave2b_calibration.py`: promoted calibrated single-label component benchmark

<!-- COMPONENT_BENCHMARK_START -->
### Generated Benchmark Snapshot

This section is generated from the benchmark manifests in `data/outputs/`.
The official published component-model scores come from the untouched `2026` holdout.

#### Single-label component benchmark

- Scope: scoped benchmark on the single-label complaint subset
- Model: `text_structured_late_fusion`
- Inputs: complaint narrative text + `wave1_incident_cohort_history` structured companion features
- Text weight: `0.75`
- Final text model: `sgd`
- Calibration: `power` alpha `1.5` from `select_2025`
- Structured branch iteration: `1280`
- Holdout macro F1: `0.7454`
- Holdout top-1 accuracy: `0.8523`
- Holdout top-3 accuracy: `0.9500`
- Holdout calibration ECE: `0.0243`

#### Multi-label routing benchmark

- Scope: full kept-case complaint routing benchmark
- Model: `CatBoost MultiLabel`
- Feature set: `core_structured`
- Threshold: `0.2`
- Selected iteration: `1200`
- Holdout macro F1: `0.2285`
- Holdout micro F1: `0.4571`
- Holdout recall@3: `0.6751`
- Holdout precision@3: `0.3027`

<!-- COMPONENT_BENCHMARK_END -->

## Component Metric And Model Glossary

These definitions explain the metrics and models used in the component-model reporting.

| Metric | Meaning | Why it matters here |
| --- | --- | --- |
| Top-1 accuracy | Share of complaints where the highest-ranked predicted component is correct. | Useful for a single best component assignment. |
| Top-3 accuracy | Share of complaints where the true component appears in the three highest-ranked predictions. | Useful because a short list of likely components can still support review. |
| Precision | Share of predictions for a component that are correct. | Shows whether a component prediction is noisy. |
| Recall | Share of true cases for a component that the model retrieves. | Shows whether the model misses component patterns. |
| F1 | One score that rewards both precision and recall. | More useful than accuracy alone when classes are uneven. |
| Macro F1 | F1 calculated for each component, then averaged so every component counts equally. | Keeps common labels from hiding poor results on rare labels. |
| Micro F1 | F1 calculated across all multi-label decisions at once. | Shows overall multi-label quality across the full dataset. |
| Recall@3 | Share of true multi-label component groups recovered in the top three predictions. | Shows whether the model puts the right labels near the top. |
| Precision@3 | Share of the top three predicted component groups that are true labels. | Shows whether the top-three list contains too many wrong labels. |
| Label coverage | Share of labels receiving at least one positive prediction. | Guards against models that ignore rare component groups. |
| ECE | Expected calibration error; how far predicted confidence is from actual accuracy. | Needed when a probability may be read as a confidence score. |
| Brier score | Probability-error score; lower means the predicted probabilities fit the true labels better. | Checks the whole probability output, not just the winning label. |

| Model or method | Meaning | Why it was used here |
| --- | --- | --- |
| Most frequent baseline | Naive model that predicts the most common component label or labels. | Establishes the minimum model quality bar. |
| Logistic regression | Simple linear model that turns weighted inputs into class probabilities. | Used as a baseline; not final because CatBoost did better on structured fields and the large final text version ran too slowly. |
| SGD classifier | Linear model trained in many small update steps. | Used for complaint text because it handles large text feature tables quickly enough for repeatable runs. |
| TF-IDF | Way to turn text into numbers by emphasizing distinctive words or character patterns. | Turns complaint narratives into model-ready features without using transformers or embeddings. |
| CatBoost | Tree-based model that handles category-heavy data well. | Used for mixed structured complaint fields such as make, model, manufacturer, state, and complaint type. |
| CatBoost MultiLabel | CatBoost adapted so one complaint can receive multiple component labels. | Kept as the official multi-label model because it preserved better overall holdout performance than the text multi-label model. |
| Late fusion | Combines scores from separately trained text and structured models. | Let the single-label model use narrative text while retaining structured vehicle/context signal. |
| Power calibration | Adjusts probabilities to make confidence scores better match observed accuracy without changing prediction order. | Corrected the single-label text-fusion confidence scores while preserving top-k accuracy. |

## Repository Structure

High-level layout:

```text
NHTSA-ODI-COMPLAINT-ANALYTICS/
|-- README.md
|-- requirements.txt
|-- pyproject.toml
|-- .gitignore
|-- .gitattributes
|-- .github/
|   |-- CODEOWNERS
|   |-- pull_request_template.md
|   `-- workflows/
|       `-- pr-ci.yml
|-- .vscode/
|   |-- extensions.json
|   `-- settings.json
|-- docs/
|   |-- CMPL.txt
|   |-- RCL.txt
|   |-- figures/
|   `-- ...
|-- data/
|   |-- raw/         # committed source zips + checksum manifest
|   |-- extracted/   # local extracted txt/csv files (ignored)
|   |-- processed/   # local parquet/csv outputs (ignored)
|   `-- outputs/     # local run summaries/manifests
|-- scripts/
|   |-- setup_env_windows.ps1
|   |-- setup_env_mac_linux.sh
|   |-- run_pipeline_windows.ps1
|   |-- run_pipeline_mac_linux.sh
|   |-- verify_install.py
|   |-- install_git_filters.py
|   `-- git_notebook_filter.py
|-- notebooks/
|   |-- EDA.ipynb
|   |-- Cleaning.ipynb
|   `-- Component_Modeling.ipynb
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
    |   `-- clean_complaints.py
    |-- features/
    |   |-- collapse_components.py
    |   `-- component_text_sidecar.py
    |-- modeling/
    |   |-- component_catboost.py
    |   |-- component_multilabel.py
    |   |-- component_text_wave2.py
    |   |-- component_text_wave2b_calibration.py
    |   `-- tune_component_catboost.py
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
- Also marks selected exploration notebooks for automatic output clearing on commit

### `.vscode/` (team editor defaults)

`.vscode/extensions.json`

- Recommends shared VS Code extensions for the team
- Most teammates will only use this indirectly by accepting recommendations

`.vscode/settings.json`

- Workspace-level VS Code settings for consistent behavior (format on save, search excludes, Python defaults)
- Windows interpreter path is preconfigured for `.venv\\Scripts\\python.exe`
- macOS/Linux teammates should select `.venv/bin/python` manually

### `.github/` (repo guardrails and review workflow)

`.github/CODEOWNERS`

- Assigns the team lead as the default code owner across the repo
- Supports required code-owner review

`.github/pull_request_template.md`

- Gives teammates a standard PR checklist
- Reminds everyone to avoid generated outputs and handle raw-data changes carefully
- Not strictly necessary for minor pull requests such as continued exploration updates, those can be commit messages

`.github/workflows/pr-ci.yml`

- GitHub Actions workflow that runs lightweight repo checks on PRs and on pushes to `main`
- Installs dependencies, runs setup verification, checks raw-data hashes/notebook hygiene, and compiles Python files

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
- `SHA256SUMS.txt` stores the approved hashes for committed raw source files

`data/extracted/`

- Local extracted files produced from raw zips (written directly into this folder)
- Generated by the pipeline and safe to delete/rebuild
- Ignored by Git

`data/processed/`

- Local pandas-friendly outputs (parquet preferred, CSV fallback)
- These are the files your EDA/modeling work will usually read
- Current key outputs include:
  - `odi_complaints_combined.parquet`
  - `odi_complaints_cleaned.parquet`
  - `odi_severity_cases.parquet`
  - `odi_component_rows.parquet`
  - `odi_component_model_cases.parquet`
  - `odi_component_multilabel_cases.parquet`
  - `odi_component_text_sidecar.parquet`
- Ignored by Git

`data/outputs/`

- Run manifests, summaries, diagnostics, and model benchmark artifacts
- Useful for verifying what a pipeline run produced
- Current examples include:
  - `clean_complaints_summary.csv`
  - `collapse_components_summary.csv`
  - `component_single_label_selection_manifest.json`
  - `component_single_label_benchmark_metrics.csv`
  - `component_textwave2b_calibration_manifest.json`
  - `component_official_benchmark_summary.csv`
  - `component_single_label_feature_importance.csv`
  - `component_multilabel_metrics.csv`
- Mostly ignored by Git, except selected official summary artifacts such as the component benchmark summary files

`docs/figures/component_models/`

- Presentation-ready component-model figures generated from saved benchmark artifacts
- Current generated figures include model lift, per-class F1, calibration, confusion, routing performance, and target-scope framing

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

`scripts/install_git_filters.py`

- Installs the repo's local Git filters after environment setup
- Enables automatic output clearing on commit for exploration notebooks listed in `.gitattributes`

`scripts/git_notebook_filter.py`

- Lightweight Git clean/smudge filter used for selected exploration notebooks
- Leaves report/presentation notebooks untouched unless you explicitly mark them in `.gitattributes`

`scripts/check_repo_integrity.py`

- Repo hygiene script used locally and in GitHub Actions
- Verifies committed raw zip hashes against `data/raw/SHA256SUMS.txt`
- Checks that exploration notebooks marked in `.gitattributes` are committed with cleared outputs

`scripts/run_pipeline_windows.ps1`

- Windows pipeline runner
- Runs verification and then the complaint extraction/preprocessing pipeline

`scripts/run_pipeline_mac_linux.sh`

- macOS/Linux pipeline runner
- Runs verification and then the complaint extraction/preprocessing pipeline

### `notebooks/` (interactive, cell-by-cell exploration)

`notebooks/EDA.ipynb`

- Main exploratory notebook for structural audit, missingness review, anomaly checks, and visual EDA

`notebooks/Cleaning.ipynb`

- Decision notebook for cleaning logic, issue flags, date handling, component grouping, and vehicle-first modeling choices

`notebooks/Component_Modeling.ipynb`

- Exploratory notebook for the original single-label structured benchmark
- Useful for diagnosis and ideas, but not the source of truth for published benchmark numbers

### `src/` ("source" folder, contains main Python files grouped by objective)

`src/__init__.py` and `src/<package>/__init__.py`

- These files mark folders as Python packages so imports work cleanly
  - i.e. the Python files in the folder can be imported the same way you would a library like Pandas with the functions in the file acting like the modules
- Example import: `from src.config.paths import ensure_project_directories`
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
- `clean_complaints.py`: conservative master cleaning plus task-specific cleaned outputs

`src/features/`

- Feature engineering code for ML/NLP/time-based models
- `collapse_components.py`: builds the single-label benchmark case table and the broader multi-label routing table
- `component_text_sidecar.py`: builds one narrative text sidecar row per complaint for component text modeling

`src/modeling/`

- Model training logic (baselines first, then stronger models)
- `component_catboost.py`: structured single-label benchmark with baselines, holdout scoring, calibration, and manifests
- `tune_component_catboost.py`: single-label feature-set selection plus focused CatBoost tuning
- `component_multilabel.py`: multi-label routing benchmark for the full kept-case target
- `component_text_wave2.py`: Wave 2 text-feature component modeling experiment path
- `component_text_wave2b_calibration.py`: calibrated single-label component model promoted as the official single-label benchmark

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
- `update_component_readme.py`: refreshes the generated component benchmark block and official summary artifacts
- `component_visuals.py`: generates presentation-ready figures for the locked component-model results

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

The setup scripts try to automate Python detection/installation, virtual environment creation, dependency install, setup verification, and local Git filter setup for exploration notebooks. Use the terminal pane (defaults to bottom of window) to run the scripts. If there is no terminal window, select Terminal -> New Terminal from the toolbar in the top-left corner.

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

- Windows: <https://www.python.org/ftp/python/3.13.12/python-3.13.12-amd64.exe>
- macOS: <https://www.python.org/ftp/python/3.13.12/python-3.13.12-macos11.pkg>

### 5) Run setup verification explicitly (optional)

The setup scripts already run verification, but you can rerun it manually anytime:

#### Windows

```powershell
.\.venv\Scripts\python.exe scripts\verify_install.py
```

If you skipped the setup script and created the environment manually, run this once to enable automatic output clearing for exploration notebooks:

```powershell
.\.venv\Scripts\python.exe scripts\install_git_filters.py
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

If you skipped the setup script and created the environment manually, run this once to enable automatic output clearing for exploration notebooks:

```bash
.venv/bin/python scripts/install_git_filters.py
```

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
- raw zip integrity is tracked in `data/raw/SHA256SUMS.txt`
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
5. Run cleaning / feature table builders as needed
6. Run modeling scripts as needed

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

### What the current ingestion pipeline does

`src/data/ingest_odi.py` currently:

- discovers complaint zips in `data/raw/`
- extracts files directly to `data/extracted/`
- reads `.txt/.csv/.tsv` files into pandas
- assigns official complaint schema column names from `docs/CMPL.txt`
- applies minor preprocessing (trim strings, parse some date-like fields, coerce common model year columns)
- writes processed parquet (preferred) or CSV outputs to `data/processed/`
- optionally builds a combined complaint dataset
- writes run manifests and simple summaries to `data/outputs/`

### Run the current cleaning and modeling flow manually

After raw ingestion, the current analysis flow is:

1. shared cleaning
2. component case collapse
3. component narrative sidecar build
4. structured single-label model selection and benchmark
5. promoted single-label text calibration benchmark
6. multi-label routing benchmark

#### Shared cleaning

```powershell
.\.venv\Scripts\python.exe -m src.preprocessing.clean_complaints --output-format parquet
```

#### Component case collapse

```powershell
.\.venv\Scripts\python.exe -m src.features.collapse_components --output-format parquet
```

#### Component narrative sidecar

```powershell
.\.venv\Scripts\python.exe -m src.features.component_text_sidecar --output-format parquet
```

#### Structured single-label model selection

```powershell
.\.venv\Scripts\python.exe -m src.modeling.tune_component_catboost --task-type CPU --n-trials 40 --seed-list 42,43,44,45,46
```

For a lighter local shakeout run that starts reporting progress immediately and skips the feature sweep unless you override it:

```powershell
.\.venv\Scripts\python.exe -m src.modeling.tune_component_catboost --task-type GPU --devices 0 --quick
```

If you want a middle ground instead of the full canonical run, keep your own trial count and seeds but bypass the feature sweep explicitly:

```powershell
.\.venv\Scripts\python.exe -m src.modeling.tune_component_catboost --task-type GPU --devices 0 --feature-set core_structured --n-trials 12 --seed-list 42,43
```

#### Structured single-label holdout benchmark

```powershell
.\.venv\Scripts\python.exe -m src.modeling.component_catboost --task-type CPU
```

For GPU runtimes that reproduce the repo artifact contract:

```bash
python -m src.modeling.tune_component_catboost --task-type GPU --devices 0 --n-trials 40 --seed-list 42,43,44,45,46
python -m src.modeling.component_catboost --task-type GPU --devices 0
```

#### Promoted single-label text benchmark

The official single-label component result is the calibrated Wave 2b text + structured late-fusion model. It depends on the component text sidecar and the Wave 2 experiment manifest.

```powershell
.\.venv\Scripts\python.exe -m src.modeling.component_text_wave2 --task-type GPU --devices 0 --skip-text-plus --final-linear-model sgd
.\.venv\Scripts\python.exe -m src.modeling.component_text_wave2b_calibration --task-type GPU --devices 0 --final-linear-model sgd --alpha-grid 1,1.25,1.5,1.75,2,2.5,3,4,5,6,8
```

#### Multi-label routing benchmark

```powershell
.\.venv\Scripts\python.exe -m src.modeling.component_multilabel
```

#### Refresh generated reporting

```powershell
.\.venv\Scripts\python.exe -m src.reporting.update_component_readme
```

#### Generate component presentation figures

```powershell
.\.venv\Scripts\python.exe -m src.reporting.component_visuals
```

These commands produce the benchmark tables, manifests, official summary artifacts, generated README benchmark section, and presentation figures under `docs/figures/component_models/`.

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

Before you say "I'm done", do this:

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
9. If you add a pure exploration notebook, mark it in `.gitattributes` so outputs are auto-cleared on commit

## Repo Guardrails

These repo-side guardrails are part of the workflow:

- `CODEOWNERS` assigns the team lead as the default code owner across the repo
- PRs use a shared checklist through `.github/pull_request_template.md`
- GitHub Actions runs `.github/workflows/pr-ci.yml` on PRs and on `main`
- `scripts/check_repo_integrity.py` checks raw zip hashes and notebook hygiene
- Exploration notebooks listed in `.gitattributes` are auto-cleared on commit
- Report/presentation notebooks can keep outputs by default
