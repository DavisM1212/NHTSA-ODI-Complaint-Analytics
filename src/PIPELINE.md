# Pipeline Architecture

This file is the live source of truth for the supported `src/` pipeline. If it conflicts with `README.md`, follow this file.

## Overview

The pipeline is intentionally organized as four clear stages:

1. Data ingestion
2. Consolidated preprocessing
3. Official modeling
4. Reporting

The supported `src` pipeline writes only persisted artifacts that are either official outputs or are directly consumed downstream by notebooks, models, reports, or visuals.

---

## Stage 1: Ingestion

**Files**

- `src/data/ingest_odi.py`
- `src/data/ingest_recalls.py` (optional starter workflow)

**Reads**

- `data/raw/*.zip`

**Writes**

- `data/processed/odi_complaints_combined.parquet`
- `data/outputs/ingest_odi_manifest.csv`
- `data/outputs/ingest_recalls_extract_manifest.csv` when recall zips are extracted

**Notes**

- ODI complaint ingest prefers the expected zip names for predictable runs
- Recall ingest remains optional and is not part of the core complaint-model pipeline

---

## Stage 2: Consolidated Preprocessing

**File**

- `src/preprocessing/clean_complaints.py`

**Reads**

- `data/processed/odi_complaints_combined.parquet`

**Writes**

- `data/processed/odi_complaints_cleaned.parquet`
- `data/processed/odi_severity_cases.parquet`
- `data/processed/odi_component_single_label_cases.parquet`
- `data/processed/odi_component_multilabel_cases.parquet`
- `data/processed/odi_component_text_sidecar.parquet`

**Optional troubleshooting outputs**

These are only written when `--summary` is passed and are kept as opt-in diagnostics:

- `data/outputs/clean_complaints_summary.csv`
- `data/outputs/clean_complaints_source_era_drift.csv`
- `data/outputs/collapse_components_summary.csv`
- `data/outputs/collapse_components_conflicts.csv`
- `data/outputs/component_target_scope_summary.csv`
- `data/outputs/component_target_scope_groups.csv`
- `data/outputs/component_text_sidecar_conflicts.csv`
- `data/outputs/component_text_overlap_report.csv`

**Notes**

- All preprocessing stays in one file so cleaning, collapse, and text sidecar logic share the same audited intermediate state in memory
- Intermediate handoff parquets such as component-row staging files are intentionally not persisted
- The troubleshooting CSVs are optional diagnostics only and are not part of the default modeling or reporting contract

---

## Stage 3: Official Modeling

**Files**

- `src/modeling/component_single_text_calibrated.py`
- `src/modeling/component_multi_routing.py`
- `src/modeling/severity_urgency_model.py`
- `src/modeling/common/helpers.py`
- `src/modeling/common/text_fusion.py`

### Severity urgency official model

**Reads**

- `data/processed/odi_severity_cases.parquet`

**Writes**

- `data/outputs/severity_urgency_official_manifest.json`
- `data/outputs/severity_urgency_official_metrics.csv`
- `data/outputs/severity_urgency_official_review_budgets.csv`
- `data/outputs/severity_urgency_official_calibration.csv`

**Notes**

- The official severity target is `severity_primary_flag`
- The official severity path keeps one `dummy_prior` baseline and one tuned calibrated late-fusion urgency model
- The broader `severity_broad_flag` work remains notebook-only sensitivity analysis for now

### Single-label official model

**Reads**

- `data/processed/odi_component_single_label_cases.parquet`
- `data/processed/odi_component_text_sidecar.parquet`

**Writes**

- `data/outputs/component_single_label_official_manifest.json`
- `data/outputs/component_single_label_official_select_grid.csv`
- `data/outputs/component_single_label_official_holdout.csv`
- `data/outputs/component_single_label_official_class_metrics.csv`
- `data/outputs/component_single_label_official_confusion_major.csv`
- `data/outputs/component_single_label_official_calibration.csv`

### Multi-label official model

**Reads**

- `data/processed/odi_component_multilabel_cases.parquet`

**Writes**

- `data/outputs/component_multilabel_official_manifest.json`
- `data/outputs/component_multilabel_official_split_summary.csv`
- `data/outputs/component_multilabel_official_metrics.csv`
- `data/outputs/component_multilabel_official_label_metrics.csv`

**Notes**

- The official single-label model remains the calibrated late-fusion text plus structured model
- The official multi-label model remains the structured CatBoost routing model
- The official severity model is a tuned calibrated late-fusion urgency model on `odi_severity_cases`
- Archived experiments and their helpers live under `notebooks/archive/`

---

## Stage 4: Reporting

**Files**

- `src/reporting/component_visuals.py`
- `src/reporting/update_component_readme.py`

**Reads**

- Official Stage 3 model artifacts in `data/outputs/`
- `data/outputs/component_textwave2b_calibration_manifest.json` for the single-label lift figure

**Writes**

- `docs/figures/component_models/*.png`
- `docs/figures/component_models/component_model_figure_index.csv`
- `data/outputs/component_official_benchmark_summary.csv`
- `data/outputs/component_official_benchmark_summary.json`
- Updated `README.md` benchmark block

**Notes**

- The Wave 2b calibration manifest remains a required reporting input for the single-label lift visual
- README publishing is driven only by the official single-label and multi-label manifests

---

## Current File Layout

```txt
src/
|-- config/
|   |-- constants.py
|   |-- contracts.py
|   |-- paths.py
|   `-- settings.py
|-- data/
|   |-- ingest_odi.py
|   |-- ingest_recalls.py
|   |-- io_utils.py
|   `-- schema_checks.py
|-- preprocessing/
|   `-- clean_complaints.py
|-- modeling/
|   |-- component_single_text_calibrated.py
|   |-- component_multi_routing.py
|   |-- severity_urgency_model.py
|   `-- common/
|       |-- helpers.py
|       `-- text_fusion.py
`-- reporting/
    |-- component_visuals.py
    `-- update_component_readme.py
```

---

## Execution Order

```bash
python -m src.data.ingest_odi
python -m src.preprocessing.clean_complaints
python -m src.modeling.severity_urgency_model
python -m src.modeling.component_single_text_calibrated
python -m src.modeling.component_multi_routing
python -m src.reporting.component_visuals
python -m src.reporting.update_component_readme
```

Use `python -m src.preprocessing.clean_complaints --summary` only when the troubleshooting CSVs are needed.
