import pandas as pd
import pytest

from src.modeling.component_common import (
    FEATURE_WAVE1_SPLIT_MODE,
    apply_multilabel_threshold,
    compose_feature_manifest,
    feature_manifest,
    prep_single_label_cases,
    split_single_label_cases,
    split_single_label_cases_by_mode,
    subset_case_frame,
)


def build_case_row(case_id, label, ldate):
    return {
        'odino': str(case_id),
        'component_group': label,
        'mfr_name': 'MFR',
        'maketxt': 'MAKE',
        'modeltxt': 'MODEL',
        'state': 'NC',
        'cmpl_type': 'C',
        'drive_train': 'AWD',
        'fuel_sys': 'FI',
        'fuel_type': 'GAS',
        'trans_type': 'AUTO',
        'fire': 'N',
        'crash': 'N',
        'medical_attn': 'N',
        'vehicles_towed_yn': 'N',
        'police_rpt_yn': 'N',
        'repaired_yn': 'N',
        'yeartxt': 2020,
        'miles': 1000,
        'veh_speed': 40,
        'injured': 0,
        'lag_days_safe': 10,
        'miles_missing_flag': False,
        'veh_speed_missing_flag': False,
        'miles_zero_flag': False,
        'veh_speed_zero_flag': False,
        'faildate_trusted_flag': True,
        'flag_date_order_bad': False,
        'flag_fail_pre_model': False,
        'flag_fail_pre_model_far': False,
        'ldate': ldate
    }


def test_feature_sets_exclude_unstable_yn_fields():
    for feature_set_name in ['core_structured', 'core_plus_quality', 'core_plus_stable_incident']:
        feature_info = feature_manifest(feature_set_name)
        assert 'orig_owner_yn' not in feature_info['feature_cols']
        assert 'anti_brakes_yn' not in feature_info['feature_cols']
        assert 'cruise_cont_yn' not in feature_info['feature_cols']


def test_split_single_label_cases_rejects_unseen_holdout_labels():
    feature_info = feature_manifest('core_structured')
    df = pd.DataFrame(
        [
            build_case_row(1, 'ENGINE / COOLING', '2024-01-01'),
            build_case_row(2, 'ENGINE / COOLING', '2025-01-01'),
            build_case_row(3, 'PARKING BRAKE', '2026-01-01')
        ]
    )
    prepared = prep_single_label_cases(df, feature_info['feature_cols'])

    with pytest.raises(ValueError, match='Holdout split has unseen target labels'):
        split_single_label_cases(prepared)


def test_apply_multilabel_threshold_enforces_top1_fallback():
    proba = [
        [0.22, 0.18, 0.10],
        [0.75, 0.05, 0.02]
    ]
    pred = apply_multilabel_threshold(proba, threshold=0.5, min_positive_labels=1)

    assert pred.tolist() == [
        [1, 0, 0],
        [1, 0, 0]
    ]


def test_compose_feature_manifest_tracks_added_and_removed_columns():
    feature_info = compose_feature_manifest(
        'wave1_test_family',
        add_cols=['state_region', 'vehicle_age_bucket'],
        remove_cols=['fire']
    )

    assert feature_info['feature_set_name'] == 'wave1_test_family'
    assert feature_info['added_cols'] == ['state_region', 'vehicle_age_bucket']
    assert feature_info['removed_cols'] == ['fire']
    assert 'state_region' in feature_info['feature_cols']
    assert 'vehicle_age_bucket' in feature_info['feature_cols']
    assert 'fire' not in feature_info['feature_cols']


def test_split_single_label_cases_by_mode_builds_feature_wave_frames():
    feature_info = compose_feature_manifest(
        'wave1_geo_family',
        add_cols=['complaint_year', 'complaint_month', 'complaint_quarter', 'vehicle_age_years', 'vehicle_age_bucket', 'state_region']
    )
    df = pd.DataFrame(
        [
            build_case_row(1, 'ENGINE / COOLING', '2023-01-01'),
            build_case_row(2, 'ENGINE / COOLING', '2024-01-01'),
            build_case_row(3, 'ENGINE / COOLING', '2025-01-01'),
            build_case_row(4, 'ENGINE / COOLING', '2026-01-01')
        ]
    )
    df['complaint_year'] = [2023, 2024, 2025, 2026]
    df['complaint_month'] = [1, 1, 1, 1]
    df['complaint_quarter'] = [1, 1, 1, 1]
    df['vehicle_age_years'] = [3, 4, 5, 6]
    df['vehicle_age_bucket'] = ['AGE_1_3', 'AGE_4_7', 'AGE_4_7', 'AGE_4_7']
    df['state_region'] = ['SOUTH', 'SOUTH', 'SOUTH', 'SOUTH']

    prepared = prep_single_label_cases(df, feature_info['feature_cols'])
    split_parts = split_single_label_cases_by_mode(prepared, split_mode=FEATURE_WAVE1_SPLIT_MODE)

    assert len(split_parts['train_core']) == 1
    assert len(split_parts['screen_2024']) == 1
    assert len(split_parts['select_2025']) == 1
    assert len(split_parts['holdout_2026']) == 1
    assert len(split_parts['dev_2020_2024']) == 2
    assert len(split_parts['dev_2020_2025']) == 3


def test_subset_case_frame_keeps_only_requested_columns():
    feature_info = compose_feature_manifest(
        'wave1_geo_family',
        add_cols=['state_region', 'complaint_year']
    )
    df = pd.DataFrame([build_case_row(1, 'ENGINE / COOLING', '2024-01-01')])
    df['state_region'] = ['SOUTH']
    df['complaint_year'] = [2024]
    df['extra_col'] = ['ignore_me']

    prepared = prep_single_label_cases(df, feature_info['feature_cols'])
    subset = subset_case_frame(prepared, ['mfr_name', 'state_region', 'complaint_year'])

    assert subset.columns.tolist() == [
        'odino',
        'ldate',
        'component_group',
        'mfr_name',
        'state_region',
        'complaint_year'
    ]
