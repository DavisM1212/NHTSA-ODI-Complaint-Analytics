import pandas as pd

# Many of these are optional for extra reporting and audit outputs, but defining
# them here is consistent. The code producing them is generally commented out to
# avoid bloat, but they can be easily re-enabled if needed.

# -----------------------------------------------------------------------------
# Processed table stems
# -----------------------------------------------------------------------------
COMBINED_COMPLAINTS_STEM = 'odi_complaints_combined'
CLEANED_COMPLAINTS_STEM = 'odi_complaints_cleaned'
SEVERITY_CASES_STEM = 'odi_severity_cases'
COMPONENT_SINGLE_LABEL_CASES_STEM = 'odi_component_single_label_cases'
COMPONENT_MULTILABEL_CASES_STEM = 'odi_component_multilabel_cases'
COMPONENT_TEXT_SIDECAR_STEM = 'odi_component_text_sidecar'


# -----------------------------------------------------------------------------
# Ingest manifests
# -----------------------------------------------------------------------------
INGEST_ODI_MANIFEST_NAME = 'ingest_odi_manifest.csv'
INGEST_RECALLS_EXTRACT_MANIFEST_NAME = 'ingest_recalls_extract_manifest.csv'


# -----------------------------------------------------------------------------
# Optional preprocessing summaries
# -----------------------------------------------------------------------------
CLEANING_SUMMARY_NAME = 'clean_complaints_summary.csv'
CLEANING_DRIFT_NAME = 'clean_complaints_source_era_drift.csv'
COMPONENT_SUMMARY_NAME = 'collapse_components_summary.csv'
COMPONENT_CONFLICT_NAME = 'collapse_components_conflicts.csv'
COMPONENT_TARGET_SCOPE_NAME = 'component_target_scope_summary.csv'
COMPONENT_TARGET_GROUP_NAME = 'component_target_scope_groups.csv'
COMPONENT_TEXT_CONFLICT_NAME = 'component_text_sidecar_conflicts.csv'
COMPONENT_TEXT_OVERLAP_NAME = 'component_text_overlap_report.csv'


# -----------------------------------------------------------------------------
# Reporting artifacts
# -----------------------------------------------------------------------------
README_START = '<!-- COMPONENT_BENCHMARK_START -->'
README_END = '<!-- COMPONENT_BENCHMARK_END -->'

COMPONENT_SINGLE_OFFICIAL_MANIFEST = 'component_single_label_official_manifest.json'
COMPONENT_MULTI_OFFICIAL_MANIFEST = 'component_multilabel_official_manifest.json'
COMPONENT_OFFICIAL_SUMMARY_CSV = 'component_official_benchmark_summary.csv'
COMPONENT_OFFICIAL_SUMMARY_JSON = 'component_official_benchmark_summary.json'
COMPONENT_TEXT_WAVE2B_CALIBRATION_MANIFEST = 'component_textwave2b_calibration_manifest.json'

COMPONENT_SINGLE_OFFICIAL_HOLDOUT = 'component_single_label_official_holdout.csv'
COMPONENT_SINGLE_OFFICIAL_SELECT_GRID = 'component_single_label_official_select_grid.csv'
COMPONENT_SINGLE_OFFICIAL_CLASS = 'component_single_label_official_class_metrics.csv'
COMPONENT_SINGLE_OFFICIAL_CONFUSION = 'component_single_label_official_confusion_major.csv'
COMPONENT_SINGLE_OFFICIAL_CALIBRATION = 'component_single_label_official_calibration.csv'

COMPONENT_MULTI_OFFICIAL_SPLIT = 'component_multilabel_official_split_summary.csv'
COMPONENT_MULTI_OFFICIAL_METRICS = 'component_multilabel_official_metrics.csv'
COMPONENT_MULTI_OFFICIAL_LABELS = 'component_multilabel_official_label_metrics.csv'


# -----------------------------------------------------------------------------
# Split modes
# -----------------------------------------------------------------------------
BENCHMARK_SPLIT_MODE = 'benchmark_v1'
FEATURE_WAVE1_SPLIT_MODE = 'feature_wave1'


# -----------------------------------------------------------------------------
# Stable data-window anchors
# -----------------------------------------------------------------------------
TRAIN_END = pd.Timestamp('2024-12-31')
VALID_END = pd.Timestamp('2025-12-31')
TRAIN_CORE_END = pd.Timestamp('2023-12-31')
SCREEN_END = pd.Timestamp('2024-12-31')
SELECT_END = pd.Timestamp('2025-12-31')
REFERENCE_DATA_YEAR = 2026
REFERENCE_MODEL_YEAR_MAX = REFERENCE_DATA_YEAR + 1


# -----------------------------------------------------------------------------
# Split contracts
# -----------------------------------------------------------------------------
SPLIT_POLICIES = {
    BENCHMARK_SPLIT_MODE: {
        'train_end': TRAIN_END,
        'valid_end': VALID_END,
        'train_name': 'train',
        'valid_name': 'valid_2025',
        'holdout_name': 'holdout_2026',
        'selection_train_name': 'train',
        'selection_eval_name': 'valid_2025',
        'dev_name': 'dev_2020_2025',
        'holdout_policy': '2026 holdout untouched during official benchmark selection'
    },
    FEATURE_WAVE1_SPLIT_MODE: {
        'train_core_end': TRAIN_CORE_END,
        'screen_end': SCREEN_END,
        'select_end': SELECT_END,
        'train_name': 'train_core',
        'screen_name': 'screen_2024',
        'select_name': 'select_2025',
        'holdout_name': 'holdout_2026',
        'selection_train_name': 'train_core',
        'selection_eval_name': 'screen_2024',
        'select_train_name': 'dev_2020_2024',
        'dev_name': 'dev_2020_2025',
        'holdout_policy': '2026 holdout untouched during feature-family screening and promotion'
    }
}


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def get_split_policy(split_mode=BENCHMARK_SPLIT_MODE):
    if split_mode not in SPLIT_POLICIES:
        choices = ', '.join(sorted(SPLIT_POLICIES))
        raise ValueError(f'Unknown split_mode {split_mode}. Choices: {choices}')
    return SPLIT_POLICIES[split_mode]
