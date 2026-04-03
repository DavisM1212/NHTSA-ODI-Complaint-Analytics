import pandas as pd
import pytest

from src.modeling.component_common import (
    apply_multilabel_threshold,
    feature_manifest,
    prep_single_label_cases,
    split_single_label_cases,
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
