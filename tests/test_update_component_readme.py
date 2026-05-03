import json
import shutil
import uuid
from pathlib import Path

import pytest

from src.reporting.update_component_readme import (
    build_summary_rows,
    update_component_readme,
)


def build_single_manifest():
    return {
        'artifact_role': 'component_single_label_official',
        'official_model': 'text_structured_late_fusion',
        'family_name': 'text_structured_late_fusion',
        'structured_feature_set': 'wave1_incident_cohort_history',
        'text_weight': 0.75,
        'final_linear_model': 'sgd',
        'calibration_method': 'power',
        'calibration_source': 'select_2025',
        'selected_alpha': 1.5,
        'selected_iteration': 1280,
        'holdout_ece': 0.0243,
        'promotion_status': 'official',
        'reporting_ready': True,
        'official_holdout_metrics': {
            'rows': 6995,
            'macro_f1': 0.7454,
            'top_1_accuracy': 0.8523,
            'top_3_accuracy': 0.9500
        },
        'artifacts': {
            'holdout': 'single_holdout.csv'
        }
    }


def build_multi_manifest():
    return {
        'artifact_role': 'component_multilabel_official',
        'selected_model': 'CatBoost MultiLabel',
        'selected_feature_set': 'core_structured',
        'selected_threshold': 0.2,
        'selected_iteration': 1200,
        'promotion_status': 'official',
        'reporting_ready': True,
        'official_holdout_metrics': {
            'rows': 10192,
            'macro_f1': 0.2285,
            'micro_f1': 0.4571,
            'recall_at_3': 0.6751,
            'precision_at_3': 0.3027,
            'label_coverage': 0.8
        },
        'artifacts': {
            'metrics': 'multi_metrics.csv'
        }
    }


def build_severity_manifest():
    return {
        'scope': 'official severity urgency benchmark',
        'publish_status': 'official',
        'target_col': 'severity_primary_flag',
        'baseline_model_name': 'dummy_prior',
        'official_model_name': 'late_fusion_sigmoid',
        'locked_params': {
            'text_weight': 0.81
        },
        'validation_metrics': {
            'official': {
                'pr_auc': 0.8282,
                'brier_score': 0.0182,
                'recall_top_5pct': 0.7565,
                'precision_top_5pct': 0.7951
            }
        },
        'holdout_metrics': {
            'official': {
                'pr_auc': 0.8452,
                'brier_score': 0.0196,
                'recall_top_5pct': 0.7233,
                'precision_top_5pct': 0.8682
            }
        }
    }


def build_nlp_manifest():
    return {
        'scope': 'official lemma-based NLP early-warning pipeline',
        'publish_status': 'official',
        'latest_watchlist_month': '2026-02-01',
        'time_windows': {
            'development_end': '2024-12-31 00:00:00',
            'forward_start': '2025-01-01 00:00:00'
        },
        'topic_model': {
            'locked_topic_k': 20
        },
        'row_counts': {
            'watchlist_rows': 7364,
            'risk_monitor_rows': 1359,
            'recurring_large_signal_rows': 3201
        }
    }


def build_nlp_watchlist_summary():
    return """month,mfr_name,maketxt,modeltxt,yeartxt,topic_id,topic_label,component_groups,complaints,unique_states,max_component_watchlist_score,avg_topic_strength,severity_broad_rate,critical_event_near_operation_rate,highway_rate,high_speed_rate,best_signal_tier
2026-02-01,FORD MOTOR COMPANY,FORD,F-150,2017,6,Transmission shifting / gear engagement issue,POWER TRAIN,39,21,24.1001,0.51,0.0,0.05,0.18,0.28,High-confidence signal
2026-02-01,MAZDA MOTOR CORP.,MAZDA,CX-90,2024,10,Steering wheel binding / difficult turning issue,STEERING,18,14,17.8027,0.57,0.0,0.0,0.17,0.17,High-confidence signal
2026-02-01,HYUNDAI MOTOR AMERICA,HYUNDAI,IONIQ 5,2024,1,Electric power steering assist loss,ELECTRICAL SYSTEM | POWER TRAIN,7,6,17.0173,0.35,0.0,0.71,0.43,0.43,Moderate signal
"""


def test_update_component_readme_replaces_marker_block():
    tmp_path = Path('data/outputs') / f'test_readme_{uuid.uuid4().hex}'
    tmp_path.mkdir(parents=True, exist_ok=True)
    try:
        readme_path = tmp_path / 'README.md'
        readme_path.write_text(
            '\n'.join(
                [
                    '# Title',
                    '<!-- COMPONENT_BENCHMARK_START -->',
                    'old block',
                    '<!-- COMPONENT_BENCHMARK_END -->',
                    'tail'
                ]
            ),
            encoding='utf-8'
        )

        single_manifest = tmp_path / 'single.json'
        single_manifest.write_text(json.dumps(build_single_manifest()), encoding='utf-8')

        multi_manifest = tmp_path / 'multi.json'
        multi_manifest.write_text(json.dumps(build_multi_manifest()), encoding='utf-8')

        severity_manifest = tmp_path / 'severity.json'
        severity_manifest.write_text(json.dumps(build_severity_manifest()), encoding='utf-8')

        nlp_manifest = tmp_path / 'nlp.json'
        nlp_manifest.write_text(json.dumps(build_nlp_manifest()), encoding='utf-8')

        nlp_summary = tmp_path / 'nlp_watchlist_summary.csv'
        nlp_summary.write_text(build_nlp_watchlist_summary(), encoding='utf-8')

        update_component_readme(
            single_manifest_path=single_manifest,
            multi_manifest_path=multi_manifest,
            severity_manifest_path=severity_manifest,
            nlp_manifest_path=nlp_manifest,
            nlp_watchlist_summary_path=nlp_summary,
            readme_path=readme_path,
            write_summary=False
        )

        updated = readme_path.read_text(encoding='utf-8')
        assert 'Severity urgency benchmark' in updated
        assert 'severity_primary_flag' in updated
        assert 'late_fusion_sigmoid' in updated
        assert '0.8282' in updated
        assert 'text_structured_late_fusion' in updated
        assert '0.7454' in updated
        assert '0.4571' in updated
        assert 'CatBoost MultiLabel' in updated
        assert '1200' in updated
        assert 'NLP early-warning snapshot' in updated
        assert 'official lemma-based NLP early-warning pipeline' in updated
        assert 'Locked topic count: `20`' in updated
        assert 'FORD F-150 2017' in updated
        assert 'Transmission shifting / gear engagement issue' in updated
        assert 'old block' not in updated
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_update_component_readme_requires_reporting_ready_manifests():
    tmp_path = Path('data/outputs') / f'test_readme_{uuid.uuid4().hex}'
    tmp_path.mkdir(parents=True, exist_ok=True)
    try:
        readme_path = tmp_path / 'README.md'
        readme_path.write_text(
            '\n'.join(
                [
                    '# Title',
                    '<!-- COMPONENT_BENCHMARK_START -->',
                    'old block',
                    '<!-- COMPONENT_BENCHMARK_END -->',
                    'tail'
                ]
            ),
            encoding='utf-8'
        )

        single_payload = build_single_manifest()
        single_payload['reporting_ready'] = False
        single_manifest = tmp_path / 'single.json'
        single_manifest.write_text(json.dumps(single_payload), encoding='utf-8')

        multi_manifest = tmp_path / 'multi.json'
        multi_manifest.write_text(json.dumps(build_multi_manifest()), encoding='utf-8')

        with pytest.raises(ValueError, match='reporting_ready'):
            update_component_readme(
                single_manifest_path=single_manifest,
                multi_manifest_path=multi_manifest,
                readme_path=readme_path,
                write_summary=False
            )
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_build_summary_rows_supports_official_manifests():
    rows = build_summary_rows(
        single_manifest=build_single_manifest(),
        multi_manifest=build_multi_manifest()
    )

    assert rows[0]['official_model'] == 'text_structured_late_fusion'
    assert rows[0]['status'] == 'official'
    assert rows[0]['ece'] == 0.0243
    assert rows[1]['official_model'] == 'CatBoost MultiLabel'
    assert rows[1]['label_coverage'] == 0.8
