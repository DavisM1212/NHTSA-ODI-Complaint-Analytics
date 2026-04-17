import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from notebooks.archive.component_single_structured_baseline import (
    build_logistic_pipeline,
    fit_histgb_stage,
)
from src.modeling.common.helpers import feature_manifest


def test_build_logistic_pipeline_matches_installed_sklearn_signature():
    feature_info = feature_manifest('core_structured')
    pipeline = build_logistic_pipeline(feature_info, random_seed=42)
    model = pipeline.named_steps['model']

    assert isinstance(model, LogisticRegression)


def test_histgb_pipeline_handles_high_cardinality_categories(monkeypatch):
    monkeypatch.setenv('OMP_NUM_THREADS', '1')
    monkeypatch.setenv('LOKY_MAX_CPU_COUNT', '1')

    feature_info = feature_manifest('core_structured')
    row_count = 300
    labels = ['ENGINE / COOLING', 'SERVICE BRAKES', 'POWER TRAIN']
    df = pd.DataFrame(
        {
            'mfr_name': [f'MFR_{idx:03d}' for idx in range(row_count)],
            'maketxt': [f'MAKE_{idx % 25:02d}' for idx in range(row_count)],
            'modeltxt': [f'MODEL_{idx % 40:02d}' for idx in range(row_count)],
            'state': ['NC'] * row_count,
            'cmpl_type': ['C'] * row_count,
            'drive_train': ['AWD'] * row_count,
            'fuel_sys': ['FI'] * row_count,
            'fuel_type': ['GAS'] * row_count,
            'trans_type': ['AUTO'] * row_count,
            'fire': ['N'] * row_count,
            'crash': ['N'] * row_count,
            'medical_attn': ['N'] * row_count,
            'vehicles_towed_yn': ['N'] * row_count,
            'yeartxt': [2020 + (idx % 4) for idx in range(row_count)],
            'miles': [1000 + idx for idx in range(row_count)],
            'veh_speed': [30 + (idx % 40) for idx in range(row_count)],
            'injured': [idx % 2 for idx in range(row_count)],
            'lag_days_safe': [idx % 365 for idx in range(row_count)],
            'miles_missing_flag': [False] * row_count,
            'veh_speed_missing_flag': [False] * row_count,
            'miles_zero_flag': [False] * row_count,
            'veh_speed_zero_flag': [False] * row_count,
            'component_group': [labels[idx % len(labels)] for idx in range(row_count)]
        }
    )

    try:
        metric_rows = fit_histgb_stage(
            train_df=df.iloc[:200].reset_index(drop=True),
            eval_df=df.iloc[200:].reset_index(drop=True),
            feature_info=feature_info,
            stage_name='unit_test',
            split_name='holdout_2026',
            random_seed=42
        )
    except PermissionError as exc:
        pytest.skip(f'HistGradientBoosting sandbox pipe restriction: {exc}')

    assert len(metric_rows) == 1
    assert metric_rows[0]['model'] == 'HistGradientBoosting'
