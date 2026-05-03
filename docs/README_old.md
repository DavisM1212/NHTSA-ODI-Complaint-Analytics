## This is the old README setup, kept for reference if needed due to the more in-depth information contained

# NHTSA ODI Complaint Analytics

Professional-grade data science workspace for analyzing National Highway Traffic Safety Administration (NHTSA) Office of Defects Investigation (ODI) consumer complaint data, with a focus on reproducible workflows, explainable analyses, and leakage-aware modeling for vehicle safety signal detection.

The repository is organized around a locked final modeling and reporting surface for ODI complaint triage, component routing, and NLP early-warning signal monitoring.

The current workflow already covers complaint ingestion, EDA, audited cleaning, lean shared complaint contracts, an official severity urgency pipeline, single-label and multi-label component target construction, one official component reporting pipeline, and an official NLP early-warning pipeline for cohort-level emerging safety signals.

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

The repo has stable, production-ready pipelines for component-level complaint routing, severity urgency scoring, and NLP early-warning watchlists. The locked final repo surface is the official severity urgency benchmark, the single-label and multi-label component benchmarks, and the official lemma-based NLP early-warning watchlist pipeline. The workflow is organized into four clear stages:

### Stage 1: Ingestion

- `src/data/ingest_odi.py`: Extract and validate ODI complaints from raw ZIP files
- Output: `data/processed/odi_complaints_combined.*`

### Stage 2: Cleaning + Features

- `src/preprocessing/clean_complaints.py`: Consolidated cleaning, severity table creation, component case collapse, and text sidecar creation
- Outputs: cleaned complaints, severity cases, single-label cases, multi-label cases, and text sidecar

### Stage 3: Main Models

- `src/modeling/severity_urgency_model.py`: Official severity urgency benchmark (tuned text + structured late fusion)
- `src/modeling/component_single_text_calibrated.py`: Official single-label component router (text + structured, late fusion)
- `src/modeling/component_multi_routing.py`: Official multi-label component router (structured only)
- `src/modeling/nlp_early_warning_system.py`: Official lemma-based NLP early-warning watchlist pipeline
- Outputs: official manifests, metrics, calibration, holdout predictions, topic libraries, watchlists, and companion tables

### Stage 4: Reporting

- `src/reporting/component_visuals.py`: Generate component confusion matrices, calibration plots, and benchmark visuals
- `src/reporting/severity_visuals.py`: Generate severity benchmark, calibration, and review-budget visuals
- `src/reporting/watchlist_visuals.py`: Generate official NLP early-warning watchlist figures
- `src/reporting/update_component_readme.py`: Update the generated README benchmark snapshot and component summary artifacts
- Outputs: `docs/figures/component_models/*.png`, `docs/figures/component_models/component_model_figure_index.csv`, `docs/figures/severity_model/*.png`, `docs/figures/severity_model/severity_model_figure_index.csv`, `docs/figures/nlp_early_warning/*.png`, `docs/figures/nlp_early_warning/nlp_early_warning_figure_index.csv`, `data/outputs/component_official_benchmark_summary.csv`, `data/outputs/component_official_benchmark_summary.json`, and the updated `README.md` benchmark block
- Reporting keeps the Wave 2b calibration manifest as an input for the single-label lift figure

### Main notebooks

- `notebooks/EDA.ipynb`: Structural audit, missingness review, anomaly checks, and visual data review
- `notebooks/Cleaning.ipynb`: Cleaning decisions, issue flags, date handling, and target-construction review
- `notebooks/Component_Modeling.ipynb`: Original component benchmark development notebook and supporting diagnosis surface
- `notebooks/Severity_Ranking_Framework.ipynb`: Severity-model development, review-budget analysis, and sensitivity review notebook
- `notebooks/NLP_Early_Warning_Framework.ipynb`: Topic-library review, watchlist interpretation, and NLP companion-table inspection notebook

These notebooks were the project's analysis, development, and review surfaces. The hardened `src/` pipelines and generated artifacts are the source of truth for official outputs.

**See [src/PIPELINE.md](src/PIPELINE.md) for the live pipeline contract and execution instructions.**

<!-- COMPONENT_BENCHMARK_START -->
### Generated Benchmark Snapshot

This section is generated from the official severity, component, and NLP early-warning artifacts in `data/outputs/`.
Severity reports the locked primary-target urgency rule on `valid_2025` plus the `2026` reference check.
The published component-model scores come from the untouched `2026` holdout.

#### Severity urgency benchmark

- Scope: official complaint-level severity urgency benchmark
- Target: `severity_primary_flag`
- Baseline: `dummy_prior`
- Model: `late_fusion_sigmoid`
- Text weight: `0.81`
- Validation PR-AUC / Brier: `0.8282` / `0.0182`
- Validation top-5% recall / precision: `0.7565` / `0.7951`
- Holdout PR-AUC / Brier: `0.8452` / `0.0196`
- Holdout top-5% recall / precision: `0.7233` / `0.8682`

#### Single-label component benchmark

- Scope: official single-label component complaint benchmark
- Model: `text_structured_late_fusion`
- Inputs: complaint narrative text + `wave1_incident_cohort_history` structured companion features
- Text weight: `0.75`
- Final text model: `sgd`
- Calibration: `power` alpha `1.5` from `select_2025`
- Structured branch iteration: `1800`
- Holdout macro F1: `0.7466`
- Holdout top-1 accuracy: `0.8522`
- Holdout top-3 accuracy: `0.9481`
- Holdout calibration ECE: `0.0251`
- Release status: `official`

#### Multi-label routing benchmark

- Scope: official multi-label complaint routing benchmark
- Model: `CatBoost MultiLabel`
- Feature set: `core_structured`
- Threshold: `0.2`
- Selected iteration: `1200`
- Holdout macro F1: `0.2285`
- Holdout micro F1: `0.4571`
- Holdout recall@3: `0.6751`
- Holdout precision@3: `0.3027`
- Release status: `official`

#### NLP early-warning snapshot

- Scope: `official lemma-based NLP early-warning pipeline`
- Locked topic count: `20`
- Development window end: `2024-12`
- Forward watchlist window start: `2025-01`
- Latest watchlist month: `2026-02`
- Watchlist rows: `7364`
- Risk monitor rows: `1359`
- Recurring large-signal rows: `3201`

Latest-month signal examples:

- `FORD F-150 2017` | `Transmission shifting / gear engagement issue` | `39` complaints | `High-confidence signal`
- `MAZDA CX-90 2024` | `Steering wheel binding / difficult turning issue` | `18` complaints | `High-confidence signal`
- `HYUNDAI IONIQ 5 2024` | `Electric power steering assist loss` | `7` complaints | `Moderate signal`

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

## Notebook Inputs And Companion Outputs

The five main notebooks are peer development and review surfaces. They sit on top of the same cleaned complaint tables and, where relevant, inspect the official modeling and reporting artifacts.

Common processed inputs across the main notebooks include:

- `data/processed/odi_complaints_cleaned.parquet`
- `data/processed/odi_severity_cases.parquet`
- `data/processed/odi_component_single_label_cases.parquet`
- `data/processed/odi_component_multilabel_cases.parquet`
- `data/processed/odi_component_text_sidecar.parquet`
- `data/processed/odi_nlp_prepped.parquet` when the official NLP cache is available

Official artifacts most often reviewed in notebooks include:

- severity outputs under `data/outputs/severity_urgency_official_*.csv` and `.json`
- component outputs under `data/outputs/component_*_official_*.csv` and `.json`
- NLP outputs under `data/outputs/nlp_early_warning_*.csv`, `.json`, and `.parquet`
- figure sets under `docs/figures/component_models/`, `docs/figures/severity_model/`, and `docs/figures/nlp_early_warning/`

Notebook-specific companion outputs still kept in the repo contract include:

- `data/outputs/severity_partner_results.csv`
- `data/outputs/severity_broad_sensitivity.csv`
- optional notebook-export CSVs under `notebooks/` for ad hoc review tables

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
|   `-- figures/
|-- data/
|   |-- raw/         # committed source zips + checksum manifest
|   |-- extracted/   # local extracted txt/csv files (ignored)
|   |-- processed/   # local parquet/csv outputs (ignored)
|   `-- outputs/     # local run summaries/manifests
|-- scripts/
|   |-- setup_env_windows.ps1
|   |-- setup_env_mac_linux.sh
|   |-- run_ingest_windows.ps1
|   |-- run_ingest_mac_linux.sh
|   |-- run_severity_official_windows.ps1
|   |-- run_severity_official_mac_linux.sh
|   |-- run_component_official_windows.ps1
|   |-- run_component_official_mac_linux.sh
|   |-- run_nlp_official_windows.ps1
|   |-- run_nlp_official_mac_linux.sh
|   |-- verify_install.py
|   |-- install_git_filters.py
|   `-- git_notebook_filter.py
|-- notebooks/
|   |-- EDA.ipynb
|   |-- Cleaning.ipynb
|   |-- Component_Modeling.ipynb
|   |-- Severity_Ranking_Framework.ipynb
|   |-- NLP_Early_Warning_Framework.ipynb
|   `-- archive/
|       |-- component_feature_wave1.py
|       |-- component_single_structured_baseline.py
|       |-- component_single_structured_tuning.py
|       |-- component_text_wave2.py
|       `-- tuning_shared.py
`-- src/
    |-- config/
    |   |-- paths.py
    |   |-- constants.py
    |   |-- contracts.py
    |   `-- settings.py
    |-- data/
    |   |-- ingest_odi.py
    |   |-- ingest_recalls.py
    |   |-- schema_checks.py
    |   `-- io_utils.py
    |-- preprocessing/
    |   `-- clean_complaints.py
    |-- modeling/
    |   |-- severity_urgency_model.py
    |   |-- component_single_text_calibrated.py
    |   |-- component_multi_routing.py
    |   |-- nlp_early_warning_system.py
    |   `-- common/
    |       |-- helpers.py
    |       `-- text_fusion.py
    `-- reporting/
        |-- component_visuals.py
        |-- severity_visuals.py
        |-- watchlist_visuals.py
        `-- update_component_readme.py
```

## Repository File/Folder Quick Guide

### `.github/` (repo automation and review workflow)

`.github/CODEOWNERS`

- Assigns the default code owner across the repo

`.github/pull_request_template.md`

- Pull request checklist for repo changes

`.github/workflows/pr-ci.yml`

- GitHub Actions workflow that runs lightweight repo checks on PRs and on pushes to `main`
- Installs dependencies, runs repo verification, checks raw-data hashes/notebook hygiene, and compiles Python files

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
  - `odi_complaints_cleaned.parquet`
  - `odi_severity_cases.parquet`
  - `odi_component_single_label_cases.parquet`
  - `odi_component_text_sidecar.parquet`
- Ignored by Git

`data/outputs/`

- Official artifacts plus downstream-supporting manifests, summaries, diagnostics, and benchmark artifacts
- Useful for verifying what a pipeline run produced
- Current examples include:
  - `component_single_label_official_manifest.json`
  - `component_multilabel_official_manifest.json`
  - `component_official_benchmark_summary.csv`
  - `severity_urgency_official_manifest.json`
- Partially ignored by Git, except selected official summary artifacts such as the component benchmark summary files

`docs/figures/component_models/`

- Presentation-ready component-model figures generated from saved benchmark artifacts
- Current generated figures include model lift, per-class F1, calibration, confusion, routing performance, and target-scope framing

`docs/figures/severity_model/`

- Presentation-ready severity-model figures generated from the official severity artifacts
- Current generated figures include split context, baseline-vs-official benchmark comparisons, review-budget tradeoffs, captured severe cases by budget, and calibration checks

### `scripts/` (quick setup commands)

`scripts/setup_env_windows.ps1` and `scripts/setup_env_mac_linux.sh`

- Repo setup automation for Windows and macOS/Linux
- Tries to install Python 3.13, creates `.venv`, installs requirements, runs verification

`scripts/verify_install.py`

- Setup diagnostic script
- Checks Python version, imports, required folders, raw zip presence, and write access

`scripts/check_repo_integrity.py`

- Repo hygiene script used locally and in GitHub Actions
- Verifies committed raw zip hashes against `data/raw/SHA256SUMS.txt`

`scripts/run_ingest_windows.ps1` and `scripts/run_ingest_mac_linux.sh`

- Windows and macOS/Linux ingest runner
- Runs verification and then the complaint extraction/combine step

`scripts/run_component_official_windows.ps1` and `scripts/run_component_official_mac_linux.sh`

- Windows and macOS/Linux official component runner
- Runs the durable component pipeline end to end: clean -> case tables -> text sidecar -> official models -> reporting

`scripts/run_severity_official_windows.ps1` and `scripts/run_severity_official_mac_linux.sh`

- Windows and macOS/Linux official severity runner
- Runs the durable severity pipeline end to end: ingest optional -> clean -> official severity model

`scripts/run_nlp_official_windows.ps1` and `scripts/run_nlp_official_mac_linux.sh`

- Windows and macOS/Linux official NLP early-warning runner
- Runs the durable NLP pipeline end to end: ingest optional -> clean -> lemma-based topic model -> official watchlist outputs -> official NLP figures

### `notebooks/` (interactive, cell-by-cell analysis and review)

`notebooks/EDA.ipynb`

- Main analysis notebook for structural audit, missingness review, anomaly checks, and visual EDA

`notebooks/Cleaning.ipynb`

- Main analysis notebook for cleaning logic, issue flags, date handling, component grouping, and vehicle-first modeling choices

`notebooks/Component_Modeling.ipynb`

- Main analysis notebook for the original single-label structured benchmark
- Useful for diagnosis and ideas, but not the source of truth for published benchmark numbers

`notebooks/Severity_Ranking_Framework.ipynb`

- Main analysis notebook for the severity-ranking section
- Loads `data/processed/odi_severity_cases.parquet`
- Documents the final urgency-rule modeling path, supporting comparisons, and the broad-target sensitivity run
- Writes `data/outputs/severity_partner_results.csv`

`notebooks/NLP_Early_Warning_Framework.ipynb`

- Main analysis notebook for the official NLP early-warning system and companion comparisons
- Loads the official lemma-based watchlist outputs plus the processed component inputs when needed for deeper inspection
- Supports topic review, watchlist interpretation, and companion-table inspection without being the source of truth for the official pipeline

### `src/` ("source" folder, contains main Python files grouped by objective)

`src/config/`

- Shared project configuration and path constants
- `paths.py`: central filesystem paths (repo root, data folders, outputs)
- `constants.py`: project constants and common field-name hints
- `contracts.py`: centralized persisted artifact names, stable year anchors, and split policies
- `settings.py`: runtime options controlled by environment variables

`src/data/`

- Ingestion and data-validation utilities
- `ingest_odi.py`: current complaint extraction + preprocessing workflow
- `ingest_recalls.py`: starter placeholder for recall extraction
- `schema_checks.py`: doc-driven schema validation (parses `docs/CMPL.txt` and `docs/RCL.txt` to validate columns, types, lengths, dates, and coded values)
- `io_utils.py`: reusable zip/file IO and preprocessing helpers

`src/preprocessing/`

- Cleaning and transformation logic
- `clean_complaints.py`: conservative master cleaning plus the final persisted preprocessing outputs used by notebooks and models

`src/modeling/`

- Official model training logic plus shared helpers
- `severity_urgency_model.py`: official severity urgency benchmark with one baseline and one tuned calibrated late-fusion path
- `component_single_text_calibrated.py`: official calibrated single-label component benchmark
- `component_multi_routing.py`: official multi-label routing benchmark
- `nlp_early_warning_system.py`: official lemma-based NLP early-warning topic model and cohort watchlist pipeline
- `common/helpers.py`: shared structured-model and split helpers
- `common/text_fusion.py`: shared text and late-fusion helpers used by the official single-label path

`src/reporting/`

- Reproducible tables/figures/report outputs for presentations and writeups
- `update_component_readme.py`: refreshes the generated README benchmark snapshot and official component summary artifacts
- `component_visuals.py`: generates presentation-ready figures for the locked component-model results
- `severity_visuals.py`: generates presentation-ready figures for the locked severity urgency model
- `watchlist_visuals.py`: generates presentation-ready figures for the locked NLP early-warning watchlist outputs

## Data Setup Instructions

Make sure raw NHTSA ODI zip files are in `data/raw/`.

Authoritative schema/data dictionary references used by the project:

- complaints: `docs/CMPL.txt`
- recalls: `docs/RCL.txt`

Use these docs when proposing schema changes, writing cleaning rules, or debugging missing/renamed columns.

Preferred complaint filenames:

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

## How To Run The Project

### Official execution order

1. Run ingest if you need to rebuild complaint inputs (`run_ingest_*`)
2. Run the official pipeline you need:
   component (`run_component_official_*`), severity (`run_severity_official_*`), or NLP early warning (`run_nlp_official_*`)
3. Inspect outputs in `data/processed/`, `data/outputs/`, and `docs/figures/`
4. Use notebooks or archive experiment scripts only when you intentionally want review or exploratory work

### Run ingest only (Windows)

```powershell
.\scripts\run_ingest_windows.ps1
```

Optional flags:

```powershell
.\scripts\run_ingest_windows.ps1 -OutputFormat csv
.\scripts\run_ingest_windows.ps1 -OverwriteExtracted
```

### Run ingest only (macOS / Linux)

```bash
./scripts/run_ingest_mac_linux.sh
```

### Run the official component pipeline (Windows)

```powershell
.\scripts\run_component_official_windows.ps1
```

### Run the official severity pipeline (Windows)

```powershell
.\scripts\run_severity_official_windows.ps1
```

### Run the official NLP early-warning pipeline (Windows)

```powershell
.\scripts\run_nlp_official_windows.ps1
```

### Run the official component pipeline (macOS / Linux)

```bash
./scripts/run_component_official_mac_linux.sh
```

### Run the official severity pipeline (macOS / Linux)

```bash
./scripts/run_severity_official_mac_linux.sh
```

### Run the official NLP early-warning pipeline (macOS / Linux)

```bash
./scripts/run_nlp_official_mac_linux.sh
```

### What the ingestion step does

`src/data/ingest_odi.py` currently:

- discovers complaint zips in `data/raw/`
- extracts files directly to `data/extracted/`
- reads `.txt/.csv/.tsv` files into pandas
- assigns official complaint schema column names from `docs/CMPL.txt`
- applies minor preprocessing (trim strings, parse some date-like fields, coerce common model year columns)
- always writes the combined complaint dataset to `data/processed/`
- writes the ODI ingest manifest to `data/outputs/`

### Run the official component flow manually

After raw ingestion, the durable component flow is:

1. shared cleaning
2. official single-label model
3. official multi-label model
4. reporting refresh

#### Shared cleaning

```powershell
.\.venv\Scripts\python.exe -m src.preprocessing.clean_complaints --output-format parquet
```

This one step writes the cleaned complaints table, the severity cases table, the single-label and multi-label component case tables, and the component text sidecar. Add `--summary` only when you want the optional troubleshooting CSVs in `data/outputs/`.

#### Official single-label component model

```powershell
.\.venv\Scripts\python.exe -m src.modeling.component_single_text_calibrated --task-type CPU
```

#### Official multi-label routing model

```powershell
.\.venv\Scripts\python.exe -m src.modeling.component_multi_routing --task-type CPU
```

#### Official severity urgency model

```powershell
.\.venv\Scripts\python.exe -m src.modeling.severity_urgency_model
```

#### Official NLP early-warning pipeline

```powershell
.\.venv\Scripts\python.exe -m src.modeling.nlp_early_warning_system
```

#### Generate NLP early-warning presentation figures

```powershell
.\.venv\Scripts\python.exe -m src.reporting.watchlist_visuals
```

#### Generate component presentation figures

```powershell
.\.venv\Scripts\python.exe -m src.reporting.component_visuals
```

#### Generate severity presentation figures

```powershell
.\.venv\Scripts\python.exe -m src.reporting.severity_visuals
```

#### Refresh generated reporting

```powershell
.\.venv\Scripts\python.exe -m src.reporting.update_component_readme
```

### Run archive experiment entrypoints manually

These scripts are archived under `notebooks/archive/`. They remain useful for historical comparisons and heavier experimentation, but they are not part of the official reporting contract. Check first to see if your GPU is compatible and, if so, use `--task-type GPU` for reasonable run times.

#### Structured single-label tuning

```powershell
.\.venv\Scripts\python.exe notebooks/archive/component_single_structured_tuning.py --task-type CPU --n-trials 40 --seed-list 42,43,44,45,46
```

#### Structured single-label benchmark baseline

```powershell
.\.venv\Scripts\python.exe notebooks/archive/component_single_structured_baseline.py --task-type CPU
```

#### Wave 1 structured feature sweep

```powershell
.\.venv\Scripts\python.exe notebooks/archive/component_feature_wave1.py --task-type GPU --devices 0
```

#### Wave 2 text and fusion exploration

```powershell
.\.venv\Scripts\python.exe notebooks/archive/component_text_wave2.py --task-type GPU --devices 0 --skip-text-plus --final-linear-model sgd
```

The official pipeline commands above produce the benchmark tables, manifests, README summary artifacts, and presentation figures under `docs/figures/component_models/`, `docs/figures/severity_model/`, and `docs/figures/nlp_early_warning/`. The archive scripts write their own exploratory manifests and tables, but those are not part of the supported reporting surface.
