import shutil
from pathlib import Path

import pandas as pd
import pytest

from src.config.contracts import (
    NLP_EARLY_WARNING_COMPLAINT_TOPICS,
    NLP_EARLY_WARNING_OFFICIAL_MANIFEST,
    NLP_EARLY_WARNING_RECURRING_SIGNALS,
    NLP_EARLY_WARNING_RISK_MONITOR,
    NLP_EARLY_WARNING_TERMS,
    NLP_EARLY_WARNING_TOPIC_LIBRARY,
    NLP_EARLY_WARNING_TOPIC_SCAN,
    NLP_EARLY_WARNING_WATCHLIST,
    NLP_EARLY_WARNING_WATCHLIST_SUMMARY,
    NLP_PREPPED_STEM,
)
from src.modeling import nlp_early_warning_system as nlp


def patch_small_topic_setup(monkeypatch):
    monkeypatch.setattr(nlp, 'TFIDF_MIN_DF', 2)
    monkeypatch.setattr(nlp, 'TOPIC_K_CANDIDATES', [2])
    monkeypatch.setattr(nlp, 'TOPIC_K_LOCK', 2)
    monkeypatch.setattr(
        nlp,
        'TOPIC_LABELS',
        {
            0: 'Transmission shifting / gear engagement issue',
            1: 'Power steering assist / steering control issue'
        }
    )
    monkeypatch.setattr(
        nlp,
        'TOPIC_CATEGORIES',
        {
            0: 'defect',
            1: 'defect'
        }
    )
    monkeypatch.setattr(nlp, 'DEFECT_WATCHLIST_TOPICS', [0, 1])
    monkeypatch.setattr(nlp, 'CAUTION_TOPICS', [])
    monkeypatch.setattr(nlp, 'PROCESS_TOPICS', [])


def add_case(
    multi_rows,
    sidecar_rows,
    odino,
    complaint_date,
    mfr_name,
    maketxt,
    modeltxt,
    yeartxt,
    state,
    component_groups,
    text,
    severity_broad=False,
    severity_primary=False
):
    multi_rows.append(
        {
            'odino': str(odino),
            'mfr_name': mfr_name,
            'maketxt': maketxt,
            'modeltxt': modeltxt,
            'yeartxt': yeartxt,
            'state': state,
            'ldate': pd.Timestamp(complaint_date),
            'severity_primary_flag': severity_primary,
            'severity_broad_flag': severity_broad,
            'component_groups': component_groups
        }
    )
    sidecar_rows.append(
        {
            'odino': str(odino),
            'cdescr': text,
            'cdescr_model_text': text,
            'cdescr_missing_flag': False,
            'cdescr_placeholder_flag': False,
            'cdescr_char_len': len(text),
            'cdescr_word_count': len(text.split()),
            'source_era': 'post_2021_schema_change',
            'ldate': pd.Timestamp(complaint_date)
        }
    )


def build_synthetic_nlp_inputs():
    multi_rows = []
    sidecar_rows = []
    next_id = 1000

    transmission_text = (
        'Transmission shifts hard into gear while driving on highway at 60 mph and the truck jerks badly'
    )
    steering_text = (
        'Power steering assist failed while driving on highway at 55 mph and the steering wheel was hard to turn'
    )

    ford_months = pd.date_range('2024-02-01', '2024-12-01', freq='MS')
    for complaint_month in ford_months:
        add_case(
            multi_rows,
            sidecar_rows,
            next_id,
            complaint_month,
            'FORD',
            'FORD',
            'F-150',
            2017,
            'TX',
            'POWER TRAIN|ELECTRICAL SYSTEM',
            transmission_text,
            severity_broad=True
        )
        next_id += 1

    add_case(
        multi_rows,
        sidecar_rows,
        next_id,
        '2025-01-01',
        'FORD',
        'FORD',
        'F-150',
        2017,
        'TX',
        'POWER TRAIN|ELECTRICAL SYSTEM',
        transmission_text,
        severity_broad=True
    )
    next_id += 1

    for state_code in ['TX', 'CA', 'FL', 'GA', 'NC']:
        add_case(
            multi_rows,
            sidecar_rows,
            next_id,
            '2025-02-01',
            'FORD',
            'FORD',
            'F-150',
            2017,
            state_code,
            'POWER TRAIN|ELECTRICAL SYSTEM',
            transmission_text,
            severity_broad=True
        )
        next_id += 1

    civic_months = pd.date_range('2024-01-01', '2024-12-01', freq='MS')
    for complaint_month in civic_months:
        add_case(
            multi_rows,
            sidecar_rows,
            next_id,
            complaint_month,
            'HONDA',
            'HONDA',
            'CIVIC',
            2020,
            'CA',
            'STEERING',
            steering_text
        )
        next_id += 1

    camry_months = pd.date_range('2024-08-01', '2025-02-01', freq='MS')
    for complaint_month in camry_months:
        for state_code in ['MI', 'OH', 'PA']:
            add_case(
                multi_rows,
                sidecar_rows,
                next_id,
                complaint_month,
                'TOYOTA',
                'TOYOTA',
                'CAMRY',
                2021,
                state_code,
                'STEERING',
                steering_text,
                severity_broad=True
            )
            next_id += 1

    multi_df = pd.DataFrame(multi_rows)
    sidecar_df = pd.DataFrame(sidecar_rows)
    return multi_df, sidecar_df


def test_load_spacy_model_fails_fast_when_model_is_missing(monkeypatch):
    monkeypatch.setattr(nlp, '_SPACY_NLP', None)

    def raise_missing(_model_name, disable=None):
        raise OSError('model not found')

    monkeypatch.setattr(nlp.spacy, 'load', raise_missing)

    with pytest.raises(RuntimeError, match='en_core_web_sm'):
        nlp.load_spacy_model()


def test_build_nlp_cache_keeps_lemma_text_and_sparse_recall_filter_uses_it(monkeypatch):
    patch_small_topic_setup(monkeypatch)

    def fake_lemmatize(text_series):
        return text_series.fillna('').astype(str).map(
            lambda text: 'takata not available'
            if 'takata' in text.lower()
            else text.lower()
        )

    monkeypatch.setattr(nlp, 'lemmatize_series', fake_lemmatize)

    multi_df = pd.DataFrame(
        [
            {
                'odino': '1',
                'mfr_name': 'FORD',
                'maketxt': 'FORD',
                'modeltxt': 'F-150',
                'yeartxt': 2017,
                'state': 'TX',
                'ldate': pd.Timestamp('2024-06-01'),
                'severity_primary_flag': False,
                'severity_broad_flag': True,
                'component_groups': 'POWER TRAIN'
            },
            {
                'odino': '2',
                'mfr_name': 'FORD',
                'maketxt': 'FORD',
                'modeltxt': 'F-150',
                'yeartxt': 2017,
                'state': 'TX',
                'ldate': pd.Timestamp('2024-07-01'),
                'severity_primary_flag': False,
                'severity_broad_flag': False,
                'component_groups': 'AIR BAGS'
            }
        ]
    )
    sidecar_df = pd.DataFrame(
        [
            {
                'odino': '1',
                'cdescr': 'Transmission shifts hard while driving on highway at 60 mph and jerks badly',
                'cdescr_model_text': 'Transmission shifts hard while driving on highway at 60 mph and jerks badly',
                'cdescr_missing_flag': False,
                'cdescr_placeholder_flag': False,
                'cdescr_char_len': 80,
                'cdescr_word_count': 12,
                'source_era': 'post_2021_schema_change',
                'ldate': pd.Timestamp('2024-06-01')
            },
            {
                'odino': '2',
                'cdescr': 'Takata air bags however the manufacturer had exceeded a reasonable amount of time for the recall repair and parts were not available for remedy completion',
                'cdescr_model_text': 'Takata air bags however the manufacturer had exceeded a reasonable amount of time for the recall repair and parts were not available for remedy completion',
                'cdescr_missing_flag': False,
                'cdescr_placeholder_flag': False,
                'cdescr_char_len': 150,
                'cdescr_word_count': 24,
                'source_era': 'post_2021_schema_change',
                'ldate': pd.Timestamp('2024-07-01')
            }
        ]
    )

    nlp_df = nlp.build_nlp_cache(multi_df, sidecar_df)

    assert {'nlp_text', 'nlp_text_lemma', 'topic_survivor_unigram_count', 'topic_model_exclude_flag'}.issubset(nlp_df.columns)
    recall_row = nlp_df.loc[nlp_df['odino'].eq('2')].iloc[0]

    assert recall_row['nlp_text_lemma'] == 'takata not available'
    assert bool(recall_row['recall_artifact_flag']) is True
    assert int(recall_row['topic_survivor_unigram_count']) <= nlp.SPARSE_RECALL_SURVIVOR_MAX
    assert bool(recall_row['topic_model_exclude_flag']) is True


def test_run_nlp_pipeline_writes_expected_artifacts_and_views(monkeypatch):
    patch_small_topic_setup(monkeypatch)
    monkeypatch.setattr(nlp, 'lemmatize_series', lambda text_series: text_series.fillna('').astype(str).str.lower())

    multi_df, sidecar_df = build_synthetic_nlp_inputs()
    output_root = Path.cwd() / 'data' / 'outputs'
    processed_root = Path.cwd() / 'data' / 'processed'
    output_dir = output_root / '_nlp_test_outputs'
    processed_dir = processed_root / '_nlp_test_processed'
    shutil.rmtree(output_dir, ignore_errors=True)
    shutil.rmtree(processed_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = nlp.run_nlp_early_warning_pipeline(
            multi_df,
            sidecar_df,
            output_dir=output_dir,
            processed_dir=processed_dir,
            publish_status='test',
            random_seed=42,
            multi_input_path=Path('synthetic_multilabel.parquet'),
            text_sidecar_path=Path('synthetic_sidecar.parquet')
        )

        expected_paths = [
            processed_dir / f'{NLP_PREPPED_STEM}.parquet',
            output_dir / NLP_EARLY_WARNING_OFFICIAL_MANIFEST,
            output_dir / NLP_EARLY_WARNING_TOPIC_SCAN,
            output_dir / NLP_EARLY_WARNING_TOPIC_LIBRARY,
            output_dir / NLP_EARLY_WARNING_COMPLAINT_TOPICS,
            output_dir / NLP_EARLY_WARNING_WATCHLIST,
            output_dir / NLP_EARLY_WARNING_WATCHLIST_SUMMARY,
            output_dir / NLP_EARLY_WARNING_RISK_MONITOR,
            output_dir / NLP_EARLY_WARNING_RECURRING_SIGNALS,
            output_dir / NLP_EARLY_WARNING_TERMS
        ]
        for artifact_path in expected_paths:
            assert artifact_path.exists()

        nlp_df = result['nlp_df']
        manifest = result['manifest']
        watchlist_df = result['cohort_watchlist_view']
        risk_monitor_df = result['cohort_risk_monitor_view']
        summary_df = result['cohort_watchlist_summary']
        clue_terms_df = result['clue_terms_df']

        assert {'nlp_text', 'nlp_text_lemma', 'topic_model_exclude_flag'}.issubset(nlp_df.columns)
        assert manifest['text_pipeline']['topic_text_col'] == 'nlp_text_lemma'
        assert manifest['publish_status'] == 'test'

        assert not watchlist_df.empty
        assert watchlist_df['complaints'].ge(nlp.COHORT_MIN_COMPLAINTS).all()
        assert (
            watchlist_df['maketxt'].eq('FORD')
            & watchlist_df['modeltxt'].eq('F-150')
        ).any()

        ford_summary = summary_df.loc[
            summary_df['maketxt'].eq('FORD')
            & summary_df['modeltxt'].eq('F-150')
        ].iloc[0]
        assert int(ford_summary['complaints']) == 5
        assert 'POWER TRAIN' in ford_summary['component_groups']
        assert 'ELECTRICAL SYSTEM' in ford_summary['component_groups']
        assert int(ford_summary['unique_states']) == 5

        assert not risk_monitor_df.empty
        assert (
            risk_monitor_df['maketxt'].eq('TOYOTA')
            & risk_monitor_df['modeltxt'].eq('CAMRY')
        ).any()

        assert not clue_terms_df.empty
        assert {'doc_support', 'doc_support_share', 'min_df_used'}.issubset(clue_terms_df.columns)
        assert clue_terms_df['month'].nunique() == 1
        assert clue_terms_df['month'].iloc[0] == watchlist_df['month'].max()
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
        shutil.rmtree(processed_dir, ignore_errors=True)


def test_run_nlp_pipeline_can_skip_cache_rebuild(monkeypatch):
    patch_small_topic_setup(monkeypatch)
    monkeypatch.setattr(nlp, 'lemmatize_series', lambda text_series: text_series.fillna('').astype(str).str.lower())

    multi_df, sidecar_df = build_synthetic_nlp_inputs()
    output_root = Path.cwd() / 'data' / 'outputs'
    processed_root = Path.cwd() / 'data' / 'processed'
    build_output_dir = output_root / '_nlp_cache_build_outputs'
    reuse_output_dir = output_root / '_nlp_cache_reuse_outputs'
    processed_dir = processed_root / '_nlp_cache_reuse_processed'

    for path in [build_output_dir, reuse_output_dir, processed_dir]:
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)

    try:
        nlp.run_nlp_early_warning_pipeline(
            multi_df,
            sidecar_df,
            output_dir=build_output_dir,
            processed_dir=processed_dir,
            publish_status='test',
            random_seed=42
        )

        def fail_build_cache(_multi_df, _sidecar_df):
            raise AssertionError('build_nlp_cache should not be called when skip_cache_rebuild=True')

        monkeypatch.setattr(nlp, 'build_nlp_cache', fail_build_cache)

        result = nlp.run_nlp_early_warning_pipeline(
            None,
            None,
            output_dir=reuse_output_dir,
            processed_dir=processed_dir,
            publish_status='test',
            random_seed=42,
            skip_cache_rebuild=True
        )

        assert result['manifest']['text_pipeline']['cache_status'] == 'reused_existing_cache'
        assert (reuse_output_dir / NLP_EARLY_WARNING_OFFICIAL_MANIFEST).exists()
        assert not result['cohort_watchlist_view'].empty
    finally:
        for path in [build_output_dir, reuse_output_dir, processed_dir]:
            shutil.rmtree(path, ignore_errors=True)


def test_run_nlp_pipeline_skip_cache_rebuild_requires_existing_cache(monkeypatch):
    patch_small_topic_setup(monkeypatch)

    output_root = Path.cwd() / 'data' / 'outputs'
    processed_root = Path.cwd() / 'data' / 'processed'
    output_dir = output_root / '_nlp_missing_cache_outputs'
    processed_dir = processed_root / '_nlp_missing_cache_processed'

    for path in [output_dir, processed_dir]:
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)

    try:
        with pytest.raises(ValueError, match='Cannot skip cache rebuild'):
            nlp.run_nlp_early_warning_pipeline(
                None,
                None,
                output_dir=output_dir,
                processed_dir=processed_dir,
                publish_status='test',
                random_seed=42,
                skip_cache_rebuild=True
            )
    finally:
        for path in [output_dir, processed_dir]:
            shutil.rmtree(path, ignore_errors=True)


def test_nlp_contract_names_are_locked():
    assert NLP_PREPPED_STEM == 'odi_nlp_prepped'
    assert NLP_EARLY_WARNING_OFFICIAL_MANIFEST == 'nlp_early_warning_official_manifest.json'
    assert NLP_EARLY_WARNING_TOPIC_SCAN == 'nlp_early_warning_topic_model_scan.csv'
    assert NLP_EARLY_WARNING_TOPIC_LIBRARY == 'nlp_early_warning_topic_library.csv'
    assert NLP_EARLY_WARNING_COMPLAINT_TOPICS == 'nlp_early_warning_complaint_topics.parquet'
    assert NLP_EARLY_WARNING_WATCHLIST == 'nlp_early_warning_watchlist.csv'
    assert NLP_EARLY_WARNING_WATCHLIST_SUMMARY == 'nlp_early_warning_watchlist_summary.csv'
    assert NLP_EARLY_WARNING_RISK_MONITOR == 'nlp_early_warning_risk_monitor.csv'
    assert NLP_EARLY_WARNING_RECURRING_SIGNALS == 'nlp_early_warning_recurring_large_signals.csv'
    assert NLP_EARLY_WARNING_TERMS == 'nlp_early_warning_terms.csv'
