# Pipeline Architecture

## Overview

The NHTSA ODI Complaint Analytics pipeline is organized into clear, sequential stages. Each stage produces stable outputs used by downstream stages.

---

## Pipeline Stages

### Stage 1: Ingestion

**Files**: `src/data/ingest_odi.py`, `src/data/ingest_recalls.py` (optional)

Extracts complaint and recall data from raw ZIP files.

- **Input**: `data/raw/*.zip`
- **Output**: `data/processed/odi_complaints_combined.*`
- **Key Functions**: Extract, validate schema, build canonical complaint table

---

### Stage 2: Unified Preprocessing

**Files**: `src/preprocessing/clean_complaints.py` (consolidated)

Single unified script that handles all preprocessing: data cleaning, case collapse, and text feature extraction. Previously this was split across 3 separate modules (`collapse_components.py`, `component_text_sidecar.py`) but has been consolidated for simpler execution and tighter data dependency management.

- **Input**: `data/processed/odi_complaints_combined.*`
- **Output**:
  - `data/processed/odi_complaints_cleaned.parquet` (core cleaned data)
  - `data/processed/odi_severity_cases.parquet` (severity flags for exploratory notebooks)
  - `data/processed/odi_component_single_label_cases.parquet` (single-label cases for modeling)
  - `data/processed/odi_component_multilabel_cases.parquet` (multi-label cases for modeling)
  - `data/processed/odi_component_text_sidecar.parquet` (NLP text features)
  - Optional diagnostic CSVs if `--summary` flag is used
- **Key Functions**:
  - **Cleaning Phase**: Validate dates, flag anomalies, standardize types, build audit trail
  - **Case Collapse Phase**: Collapse multi-component complaint rows into single-label and multi-label case tables
  - **Text Sidecar Phase**: Extract and normalize complaint narratives, vectorize text, handle missing values
- **CLI**: `python -m src.preprocessing.clean_complaints [--summary]`

---

### Stage 3: Main Models

**Files**:

- `src/modeling/component_single_text_calibrated.py` (Component routing: single-label)
- `src/modeling/component_multi_routing.py` (Component routing: multi-label)

Trains and evaluates official component routing models using structured + text features.

- **Input**: Cleaned data tables from Stage 2
- **Output**:
  - `data/outputs/component_single_label_official_manifest.json`
  - `data/outputs/component_single_label_official_class_metrics.csv`
  - `data/outputs/component_single_label_official_confusion_major.csv`
  - `data/outputs/component_single_label_official_calibration.csv`
  - `data/outputs/component_single_label_official_holdout.json`
  - `data/outputs/component_multilabel_official_manifest.json`
  - `data/outputs/component_multilabel_official_metrics.csv`
  - `data/outputs/component_multilabel_official_label_metrics.csv`
- **Model Details**: CatBoost for structured features + linear models for text fusion
- **Validation**: Time-aware splits (2020–2024 train, 2025 valid, 2026 holdout)

---

### Stage 4: Reporting

**Files**: `src/reporting/component_visuals.py`, `src/reporting/update_component_readme.py`

Generates visualizations and updates project README with official results.

- **Input**: Official model outputs from Stage 3
- **Output**:
  - `docs/figures/component_models/*.png`
  - Modified `README.md`
- **Key Functions**: Render confusion matrices, calibration plots, class metrics visualizations

---

## Future Stages (WIP in notebooks)

### Severity Ranking Framework

**Location**: `notebooks/Severity_Ranking_Framework.ipynb`

Will build models to prioritize complaints by severity. Not yet hardened into pipeline.

### NLP Early-Warning Framework

**Location**: `notebooks/NLP_Early_Warning_Framework.ipynb`

Will detect emerging safety issues from complaint narratives. Not yet hardened into pipeline.

---

## Shared Helpers

**Location**: `src/modeling/common/helpers.py`

Consolidated module (merged from `core.py` + `multilabel.py`) containing:

- **Constants**: `STATE_REGION_MAP`, `VEHICLE_AGE_BUCKETS`, feature definitions, split policies
- **Feature Preparation**: `add_requested_case_features`, `derive_vehicle_age_features`, `derive_state_region`
- **Data Splitting**: `split_single_label_cases_by_mode`, `split_multi_label_cases_by_mode`
- **Model Building**: `build_catboost_model`, `fit_catboost_with_external_selection`
- **Scoring & Metrics**: `score_multiclass_from_proba`, `build_multiclass_calibration_df`
- **Multilabel Utilities**: `apply_multilabel_threshold`, `select_multilabel_threshold`
- **Text Fusion**: Accessed via `src/modeling/common/text_fusion.py`

---

## File Structure Summary

```txt
src/
├── config/                      # Configuration & constants
│   ├── paths.py
│   ├── constants.py
│   ├── contracts.py             # Output artifact names
│   └── settings.py
├── data/                        # Data ingestion
│   ├── ingest_odi.py
│   ├── ingest_recalls.py
│   ├── io_utils.py
│   └── schema_checks.py
├── preprocessing/               # Stage 2: Unified preprocessing
│   └── clean_complaints.py      # Consolidated: cleaning + case collapse + text features
├── modeling/                    # Stage 3: Official models
│   ├── component_single_text_calibrated.py
│   ├── component_multi_routing.py
│   └── common/                  # Shared helpers
│       ├── helpers.py           # Consolidated core + multilabel
│       └──  text_fusion.py      # Text feature fusion workflows
│   └──official/
└── reporting/                   # Stage 4: Reporting
    ├── component_visuals.py
    └── update_component_readme.py
```

---

## Output Artifacts

### Preprocessing Outputs (Stage 2, in `data/processed/`)

**Cleaned Data Tables**:

- `odi_complaints_cleaned.parquet` (core cleaned complaint records)
- `odi_component_single_label_cases.parquet` (single-label cases for modeling)
- `odi_component_multilabel_cases.parquet` (multi-label cases for modeling)
- `odi_component_text_sidecar.parquet` (NLP features)

**Optional Diagnostic CSVs** (with `--summary` flag):

- `clean_complaints_summary.csv`
- `clean_complaints_source_era_drift.csv`
- `collapse_components_summary.csv`
- `collapse_components_conflicts.csv`
- `component_target_scope_summary.csv`
- `component_target_group_summary.csv`
- `component_text_sidecar_conflicts.csv`
- `component_text_overlap_report.csv`

### Model Outputs (Stage 3, in `data/outputs/`)

**Official Component Model Results**:

- `component_single_label_official_manifest.json`
- `component_single_label_official_class_metrics.csv`
- `component_single_label_official_confusion_major.csv`
- `component_single_label_official_calibration.csv`
- `component_single_label_official_holdout.json`
- `component_multilabel_official_manifest.json`
- `component_multilabel_official_metrics.csv`
- `component_multilabel_official_label_metrics.csv`

### Archived Outputs

Wave 1/2 experiment CSVs and CatBoost intermediate files were removed during consolidation. Experiment logic preserved in `notebooks/archive/` for reference.

---

## Execution

Run pipeline stages sequentially:

```bash
# Stage 1: Ingest
python -m src.data.ingest_odi

# Stage 2: Unified preprocessing (cleaning, case collapse, text features)
python -m src.preprocessing.clean_complaints [--summary]

# Stage 3: Train official models
python -m src.modeling.component_single_text_calibrated
python -m src.modeling.component_multi_routing

# Stage 4: Report
python -m src.reporting.component_visuals
python -m src.reporting.update_component_readme
```

**Note**: Stage 2 now runs all preprocessing (cleaning, collapse, text) in a single consolidated script. The `--summary` flag enables optional diagnostic CSVs for validation and reporting.

---

## Notes

- All raw data in `data/raw/` is treated as immutable
- Processed outputs are reproducible from pipeline code and raw data
- Model hyperparameters are documented in official model files
- Time-aware validation splits prevent leakage (train: 2020–2024, valid: 2025, holdout: 2026)
- **Consolidation (April 2026)**: Preprocessing unified into single script; eliminated ~2.6 GB of intermediate parquets; merged duplicate code modules (`core.py` + `multilabel.py` → `helpers.py`). See `CONSOLIDATION_COMPLETE.md` for full details.
