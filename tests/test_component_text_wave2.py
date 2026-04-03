import numpy as np
import pandas as pd

from src.modeling.component_text_wave2 import (
    FUSION_TEXT_WEIGHTS,
    MULTI_THRESHOLDS,
    STRUCTURED_FEATURE_SET,
    build_multi_manifest_entry,
    build_single_manifest_entry,
    build_overlap_mask,
    fit_text_vectorizers,
    merge_text_sidecar,
    prepare_text_sidecar,
    safe_multi_predict_proba,
    safe_single_predict_proba,
    select_multi_fusion_weight,
    select_single_fusion_weight,
)


def test_wave2_constants_match_the_locked_plan():
    assert STRUCTURED_FEATURE_SET == 'wave1_incident_cohort_history'
    assert FUSION_TEXT_WEIGHTS == [0.25, 0.50, 0.75]
    assert MULTI_THRESHOLDS == [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25, 0.30]


def test_merge_text_sidecar_fills_defaults_for_missing_cases():
    case_df = pd.DataFrame(
        {
            'odino': ['1', '2'],
            'component_group': ['ENGINE', 'BRAKES'],
            'ldate': ['2024-01-01', '2024-01-02'],
            'source_era': ['post_2021_case', pd.NA]
        }
    )
    sidecar_df = pd.DataFrame(
        {
            'odino': ['1'],
            'cdescr': ['engine failure'],
            'cdescr_model_text': ['engine failure'],
            'cdescr_missing_flag': [False],
            'cdescr_placeholder_flag': [False],
            'cdescr_char_len': [14],
            'cdescr_word_count': [2],
            'source_era': ['post_2021'],
            'ldate': ['2024-01-01']
        }
    )

    merged = merge_text_sidecar(case_df, sidecar_df)
    missing_row = merged.loc[merged['odino'].eq('2')].iloc[0]

    assert len(merged) == 2
    assert merged.loc[merged['odino'].eq('1'), 'source_era'].iloc[0] == 'post_2021_case'
    assert pd.isna(merged.loc[merged['odino'].eq('2'), 'source_era'].iloc[0])
    assert missing_row['cdescr_model_text'] == ''
    assert bool(missing_row['cdescr_missing_flag']) is True
    assert int(missing_row['cdescr_word_count']) == 0


def test_text_vectorizers_only_learn_from_training_text():
    vectorizers = fit_text_vectorizers(
        pd.Series(
            [
                'engine issue noise',
                'engine issue stall',
                'engine issue vibration',
                'engine issue smoke',
                'engine issue surge',
                'engine fault rough',
                'engine fault hesitation',
                'engine fault heat',
                'engine fault shake',
                'brake fault leak'
            ]
        )
    )

    assert 'moonroofzz' not in vectorizers['word'].vocabulary_
    assert 'engine' in vectorizers['word'].vocabulary_


def test_prepare_text_sidecar_rejects_duplicate_odino_rows():
    sidecar_df = pd.DataFrame(
        {
            'odino': ['1', '1'],
            'cdescr': ['a', 'b'],
            'cdescr_model_text': ['a', 'b'],
            'cdescr_missing_flag': [False, False],
            'cdescr_placeholder_flag': [False, False],
            'cdescr_char_len': [1, 1],
            'cdescr_word_count': [1, 1],
            'source_era': ['post_2021', 'post_2021'],
            'ldate': ['2024-01-01', '2024-01-02']
        }
    )

    try:
        prepare_text_sidecar(sidecar_df)
        assert False, 'Expected duplicate odino validation to fail'
    except ValueError as exc:
        assert 'duplicate odino rows' in str(exc)


def test_fusion_helpers_respect_declared_grids():
    y_single = np.array(['A', 'B', 'C', 'D'])
    text_single = np.array(
        [
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.8, 0.05, 0.05],
            [0.1, 0.2, 0.6, 0.1],
            [0.1, 0.1, 0.1, 0.7]
        ]
    )
    structured_single = np.array(
        [
            [0.6, 0.2, 0.1, 0.1],
            [0.2, 0.6, 0.1, 0.1],
            [0.2, 0.3, 0.4, 0.1],
            [0.2, 0.1, 0.2, 0.5]
        ]
    )
    single_choice = select_single_fusion_weight(
        y_single,
        text_single,
        np.array(['A', 'B', 'C', 'D']),
        structured_single,
        np.array(['A', 'B', 'C', 'D'])
    )

    y_multi = np.array([[1, 0], [0, 1]])
    text_multi = np.array([[0.8, 0.2], [0.2, 0.8]])
    structured_multi = np.array([[0.6, 0.4], [0.4, 0.6]])
    multi_choice = select_multi_fusion_weight(y_multi, text_multi, structured_multi)

    assert single_choice['selected_text_weight'] in FUSION_TEXT_WEIGHTS
    assert multi_choice['selected_text_weight'] in FUSION_TEXT_WEIGHTS
    assert multi_choice['selected_threshold'] in MULTI_THRESHOLDS


def test_build_overlap_mask_uses_exact_nonempty_matches():
    mask = build_overlap_mask(
        ['engine failed', '', 'brake issue'],
        ['engine failed', '', 'other text']
    )

    assert mask.tolist() == [True, False, False]


def test_checkpoint_manifest_entries_include_stage_metadata():
    single_entry = build_single_manifest_entry(
        {
            'input_path': 'single.parquet',
            'text_sidecar_path': 'sidecar.parquet',
            'selected_family': 'text_only_linear',
            'checkpoint_stage': 'screen_2024_complete',
            'completed_stages': ['screen_2024']
        },
        {'macro_f1': 0.1},
        {'macro_f1': 0.2},
        0.01
    )
    multi_entry = build_multi_manifest_entry(
        {
            'input_path': 'multi.parquet',
            'text_sidecar_path': 'sidecar.parquet',
            'selected_family': 'structured_carry_forward',
            'checkpoint_stage': 'select_2025_complete',
            'completed_stages': ['screen_2024', 'select_2025']
        },
        {'macro_f1': 0.1},
        {'macro_f1': 0.2}
    )

    assert single_entry['checkpoint_stage'] == 'screen_2024_complete'
    assert single_entry['artifacts']['screen'].endswith('component_single_label_textwave2_screen.csv')
    assert multi_entry['checkpoint_stage'] == 'select_2025_complete'
    assert multi_entry['artifacts']['holdout'].endswith('component_multilabel_textwave2_holdout.csv')


def test_safe_single_predict_proba_falls_back_to_decision_scores():
    class DummySingleModel:
        classes_ = np.array(['A', 'B', 'C'])

        def predict_proba(self, matrix):
            return np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]])

        def decision_function(self, matrix):
            return np.array([[10.0, 1.0, -2.0], [0.5, 3.0, 0.0]])

    proba = safe_single_predict_proba(DummySingleModel(), np.zeros((2, 3)))

    assert np.isfinite(proba).all()
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_safe_multi_predict_proba_falls_back_to_decision_scores():
    class DummyMultiModel:
        def predict_proba(self, matrix):
            return np.array([[np.nan, np.nan], [np.nan, np.nan]])

        def decision_function(self, matrix):
            return np.array([[2.0, -1.0], [0.0, 3.0]])

    proba = safe_multi_predict_proba(DummyMultiModel(), np.zeros((2, 2)))

    assert np.isfinite(proba).all()
    assert ((proba >= 0.0) & (proba <= 1.0)).all()
