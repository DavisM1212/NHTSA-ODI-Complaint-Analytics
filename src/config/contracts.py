from src.config.paths import OUTPUTS_DIR, PROCESSED_DATA_DIR

# -----------------------------------------------------------------------------
# Processed table stems
# -----------------------------------------------------------------------------
COMBINED_COMPLAINTS_STEM = 'odi_complaints_combined'
CLEANED_COMPLAINTS_STEM = 'odi_complaints_cleaned'
CLEANING_AUDIT_STEM = 'odi_complaints_cleaning_audit'
SEVERITY_CASES_STEM = 'odi_severity_cases'
COMPONENT_ROWS_STEM = 'odi_component_rows'
COMPONENT_CASE_BASE_STEM = 'odi_component_case_base'
COMPONENT_SINGLE_LABEL_CASES_STEM = 'odi_component_single_label_cases'
COMPONENT_MULTILABEL_CASES_STEM = 'odi_component_multilabel_cases'
COMPONENT_TEXT_SIDECAR_STEM = 'odi_component_text_sidecar'


# -----------------------------------------------------------------------------
# Summary and audit outputs
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
# Official reporting artifacts
# -----------------------------------------------------------------------------
README_START = '<!-- COMPONENT_BENCHMARK_START -->'
README_END = '<!-- COMPONENT_BENCHMARK_END -->'

COMPONENT_SINGLE_OFFICIAL_MANIFEST = 'component_single_label_official_manifest.json'
COMPONENT_MULTI_OFFICIAL_MANIFEST = 'component_multilabel_official_manifest.json'
COMPONENT_OFFICIAL_SUMMARY_CSV = 'component_official_benchmark_summary.csv'
COMPONENT_OFFICIAL_SUMMARY_JSON = 'component_official_benchmark_summary.json'

COMPONENT_SINGLE_OFFICIAL_HOLDOUT = 'component_single_label_official_holdout.csv'
COMPONENT_SINGLE_OFFICIAL_CLASS = 'component_single_label_official_class_metrics.csv'
COMPONENT_SINGLE_OFFICIAL_CONFUSION = 'component_single_label_official_confusion_major.csv'
COMPONENT_SINGLE_OFFICIAL_CALIBRATION = 'component_single_label_official_calibration.csv'

COMPONENT_MULTI_OFFICIAL_SPLIT = 'component_multilabel_official_split_summary.csv'
COMPONENT_MULTI_OFFICIAL_METRICS = 'component_multilabel_official_metrics.csv'
COMPONENT_MULTI_OFFICIAL_LABELS = 'component_multilabel_official_label_metrics.csv'


# -----------------------------------------------------------------------------
# Helper paths
# -----------------------------------------------------------------------------
def processed_stem_path(stem):
    return PROCESSED_DATA_DIR / stem


def output_path(name):
    return OUTPUTS_DIR / name
