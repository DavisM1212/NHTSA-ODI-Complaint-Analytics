import shutil
from pathlib import Path

import pandas as pd

from src.config.contracts import (
    SEVERITY_URGENCY_OFFICIAL_CALIBRATION,
    SEVERITY_URGENCY_OFFICIAL_MANIFEST,
    SEVERITY_URGENCY_OFFICIAL_METRICS,
    SEVERITY_URGENCY_OFFICIAL_REVIEW_BUDGETS,
)
from src.modeling.severity_urgency_model import (
    BASELINE_NAME,
    ISOTONIC_NAME,
    RAW_NAME,
    SIGMOID_NAME,
    build_custom_stop_words,
    build_review_budget_row,
    build_text_series,
    pick_calibration_winner,
    run_severity_pipeline,
)


def build_synthetic_severity_df():
    rows = []

    def add_rows(year, start_idx, labels):
        for offset, label in enumerate(labels):
            is_positive = bool(label)
            day = min(offset + 1, 28)
            rows.append(
                {
                    'odino': str(start_idx + offset),
                    'ldate': pd.Timestamp(f'{year}-0{(offset % 3) + 1}-{day:02d}'),
                    'cdescr': (
                        'The contact stated vehicle stalled while driving and had no power no acceleration'
                        if is_positive
                        else 'The contact stated radio static and paint peeling cosmetic concern only'
                    ),
                    'severity_primary_flag': is_positive,
                    'severity_broad_flag': is_positive,
                    'mfr_name': 'FORD' if is_positive else 'HONDA',
                    'maketxt': 'FORD' if is_positive else 'HONDA',
                    'modeltxt': 'ESCAPE' if is_positive else 'CIVIC',
                    'state': 'NC' if is_positive else 'CA',
                    'cmpl_type': 'V' if is_positive else 'I',
                    'drive_train': 'FWD' if is_positive else pd.NA,
                    'fuel_type': 'GS',
                    'police_rpt_yn': 'Y' if is_positive else 'N',
                    'repaired_yn': pd.NA,
                    'orig_owner_yn': 'N',
                    'yeartxt': 2018 if is_positive else 2019,
                    'miles': 45000 if is_positive else 30000,
                    'veh_speed': 60 if is_positive else 15,
                    'lag_days_safe': 10 if is_positive else 60,
                    'miles_missing_flag': False,
                    'veh_speed_missing_flag': False,
                    'faildate_trusted_flag': True,
                    'faildate_untrusted_flag': False,
                    'component_count': 2 if is_positive else 1,
                    'row_count': 1
                }
            )

    add_rows(2024, 100000, [1] * 20 + [0] * 20)
    add_rows(2025, 200000, [1] * 10 + [0] * 10)
    add_rows(2026, 300000, [1] * 10 + [0] * 10)
    return pd.DataFrame(rows)


def test_build_text_series_applies_light_and_error_cleanup():
    source_df = pd.DataFrame(
        {
            'cdescr_model_text': [
                'The contact stated [XXX] Information redacted pursuant to the Freedom of Information Act (FOIA) 5 U.S.C. 552(b)(6) and 49 C.F.R. 512.8 case #ABC12345 call 555-123-4567'
            ]
        }
    )

    cleaned = build_text_series(source_df, clean_mode='light', error_cleanup=True).iloc[0]

    assert 'the contact stated' not in cleaned.lower()
    assert '[' not in cleaned
    assert 'phone_token' in cleaned
    assert 'case_id_token' in cleaned


def test_custom_stop_words_preserve_negation_and_timing_terms():
    stop_words = set(build_custom_stop_words())

    for word in ['no', 'not', 'never', 'without', 'while', 'after', 'before']:
        assert word not in stop_words

    assert 'the' in stop_words


def test_pick_calibration_winner_prefers_sigmoid_when_brier_is_effectively_tied():
    raw_row = {
        'model': RAW_NAME,
        'brier_score': 0.0300,
        'recall_top_5pct': 0.7500
    }
    sigmoid_row = {
        'model': SIGMOID_NAME,
        'brier_score': 0.0200,
        'recall_top_5pct': 0.7490
    }
    isotonic_row = {
        'model': ISOTONIC_NAME,
        'brier_score': 0.0195,
        'recall_top_5pct': 0.7490
    }

    assert pick_calibration_winner(raw_row, sigmoid_row, isotonic_row) == SIGMOID_NAME


def test_build_review_budget_row_is_stable_under_tied_scores():
    row = build_review_budget_row(
        [True, False, True, False],
        [0.25, 0.25, 0.25, 0.25],
        0.50
    )

    assert row['flagged_rows'] == 2
    assert row['severe_cases_captured'] == 1
    assert row['precision_within_flagged_set'] == 0.5
    assert row['recall_within_flagged_set'] == 0.5


def test_run_severity_pipeline_writes_expected_artifacts():
    severity_df = build_synthetic_severity_df()
    output_root = Path.cwd() / 'data' / 'outputs'
    output_dir = output_root / '_severity_test_outputs'
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = run_severity_pipeline(
            severity_df,
            input_path=Path('synthetic_severity_cases.parquet'),
            output_dir=output_dir,
            random_seed=42,
            publish_status='test'
        )

        assert (output_dir / SEVERITY_URGENCY_OFFICIAL_MANIFEST).exists()
        assert (output_dir / SEVERITY_URGENCY_OFFICIAL_METRICS).exists()
        assert (output_dir / SEVERITY_URGENCY_OFFICIAL_REVIEW_BUDGETS).exists()
        assert (output_dir / SEVERITY_URGENCY_OFFICIAL_CALIBRATION).exists()

        metrics_df = result['metrics_df']
        review_budget_df = result['review_budget_df']
        calibration_df = result['calibration_df']
        manifest = result['manifest']

        assert set(metrics_df['split']) == {'valid_2025', 'holdout_2026'}
        assert {BASELINE_NAME, RAW_NAME, SIGMOID_NAME, ISOTONIC_NAME}.issubset(set(metrics_df['model']))
        assert metrics_df['is_official'].sum() == 2
        assert metrics_df['is_baseline'].sum() == 2

        assert set(review_budget_df['split']) == {'valid_2025', 'holdout_2026'}
        assert set(review_budget_df['model']) == {BASELINE_NAME, manifest['official_model_name']}
        assert set(review_budget_df['budget_label']) == {'top_1pct', 'top_2pct', 'top_5pct', 'top_10pct'}

        assert set(calibration_df['split']) == {'valid_2025', 'holdout_2026'}
        assert manifest['baseline_model_name'] == BASELINE_NAME
        assert manifest['official_model_name'] in {RAW_NAME, SIGMOID_NAME, ISOTONIC_NAME}
        assert manifest['publish_status'] == 'test'
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_severity_contract_names_are_locked():
    assert SEVERITY_URGENCY_OFFICIAL_MANIFEST == 'severity_urgency_official_manifest.json'
    assert SEVERITY_URGENCY_OFFICIAL_METRICS == 'severity_urgency_official_metrics.csv'
    assert SEVERITY_URGENCY_OFFICIAL_REVIEW_BUDGETS == 'severity_urgency_official_review_budgets.csv'
    assert SEVERITY_URGENCY_OFFICIAL_CALIBRATION == 'severity_urgency_official_calibration.csv'
