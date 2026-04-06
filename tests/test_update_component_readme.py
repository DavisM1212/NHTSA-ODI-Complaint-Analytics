import json
import shutil
import uuid
from pathlib import Path

from src.reporting.update_component_readme import build_summary_rows, update_component_readme


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
        single_manifest.write_text(
            json.dumps(
                {
                    'selected_feature_set': 'core_structured',
                    'selected_iteration': 123,
                    'official_holdout_metrics': {
                        'macro_f1': 0.2222,
                        'top_1_accuracy': 0.3333,
                        'top_3_accuracy': 0.5555
                    }
                }
            ),
            encoding='utf-8'
        )

        multi_manifest = tmp_path / 'multi.json'
        multi_manifest.write_text(
            json.dumps(
                {
                    'selected_model': 'CatBoost MultiLabel',
                    'selected_feature_set': 'core_structured',
                    'selected_threshold': 0.3,
                    'selected_iteration': 250,
                    'official_holdout_metrics': {
                        'macro_f1': 0.1111,
                        'micro_f1': 0.4444,
                        'recall_at_3': 0.6666,
                        'precision_at_3': 0.2222
                    }
                }
            ),
            encoding='utf-8'
        )

        update_component_readme(
            single_manifest_path=single_manifest,
            multi_manifest_path=multi_manifest,
            readme_path=readme_path,
            write_summary=False
        )

        updated = readme_path.read_text(encoding='utf-8')
        assert '0.2222' in updated
        assert '0.4444' in updated
        assert 'CatBoost MultiLabel' in updated
        assert '250' in updated
        assert 'old block' not in updated
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_update_component_readme_supports_wave2b_single_label_manifest():
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

        single_manifest = tmp_path / 'single_wave2b.json'
        single_manifest.write_text(
            json.dumps(
                {
                    'artifact_role': 'text_wave2b_calibration',
                    'family_name': 'text_structured_late_fusion',
                    'structured_feature_set': 'wave1_incident_cohort_history',
                    'text_weight': 0.75,
                    'final_linear_model': 'sgd',
                    'calibration_method': 'power',
                    'calibration_source': 'select_2025',
                    'selected_alpha': 1.5,
                    'selected_iteration': 1280,
                    'holdout_ece': 0.0243,
                    'promotion_status': 'promoted',
                    'calibrated_holdout_metrics': {
                        'rows': 6995,
                        'macro_f1': 0.7454,
                        'top_1_accuracy': 0.8523,
                        'top_3_accuracy': 0.9500
                    }
                }
            ),
            encoding='utf-8'
        )

        multi_manifest = tmp_path / 'multi.json'
        multi_manifest.write_text(
            json.dumps(
                {
                    'selected_model': 'CatBoost MultiLabel',
                    'selected_feature_set': 'core_structured',
                    'selected_threshold': 0.2,
                    'selected_iteration': 1200,
                    'official_holdout_metrics': {
                        'macro_f1': 0.2285,
                        'micro_f1': 0.4571,
                        'recall_at_3': 0.6751,
                        'precision_at_3': 0.3027
                    }
                }
            ),
            encoding='utf-8'
        )

        update_component_readme(
            single_manifest_path=single_manifest,
            multi_manifest_path=multi_manifest,
            readme_path=readme_path,
            write_summary=False
        )

        updated = readme_path.read_text(encoding='utf-8')
        assert 'text_structured_late_fusion' in updated
        assert 'power' in updated
        assert '0.7454' in updated
        assert '0.0243' in updated
        assert 'CatBoost MultiLabel' in updated
        assert 'old block' not in updated
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_build_summary_rows_supports_wave2b_and_multilabel():
    rows = build_summary_rows(
        single_manifest={
            'artifact_role': 'text_wave2b_calibration',
            'family_name': 'text_structured_late_fusion',
            'structured_feature_set': 'wave1_incident_cohort_history',
            'holdout_ece': 0.0243,
            'selected_alpha': 1.5,
            'calibration_method': 'power',
            'promotion_status': 'promoted',
            'calibrated_holdout_metrics': {
                'rows': 6995,
                'macro_f1': 0.7454,
                'top_1_accuracy': 0.8523,
                'top_3_accuracy': 0.9500
            }
        },
        multi_manifest={
            'artifact_role': 'final_benchmark',
            'selected_model': 'CatBoost MultiLabel',
            'selected_feature_set': 'core_structured',
            'official_holdout_metrics': {
                'rows': 10192,
                'macro_f1': 0.2285,
                'micro_f1': 0.4571,
                'recall_at_3': 0.6751,
                'precision_at_3': 0.3027,
                'label_coverage': 0.8
            }
        }
    )

    assert rows[0]['official_model'] == 'text_structured_late_fusion'
    assert rows[0]['status'] == 'promoted'
    assert rows[0]['ece'] == 0.0243
    assert rows[1]['official_model'] == 'CatBoost MultiLabel'
    assert rows[1]['label_coverage'] == 0.8
