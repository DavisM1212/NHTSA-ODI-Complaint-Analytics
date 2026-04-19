import json
import shutil

import pandas as pd

from src.config.paths import OUTPUTS_DIR
from src.reporting.component_visuals import generate_component_visuals


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_generate_component_visuals_writes_expected_files():
    test_root = OUTPUTS_DIR / '_component_visuals_test'
    outputs_dir = test_root / 'outputs'
    figures_dir = test_root / 'figures'
    shutil.rmtree(test_root, ignore_errors=True)
    try:
        outputs_dir.mkdir(parents=True)
        write_csv(
            outputs_dir / 'component_official_benchmark_summary.csv',
            [
                {
                    'task': 'single_label_component',
                    'official_model': 'text_structured_late_fusion',
                    'artifact_role': 'component_single_label_official',
                    'feature_set': 'wave1_incident_cohort_history',
                    'holdout_rows': 100,
                    'macro_f1': 0.75,
                    'top_1_accuracy': 0.85,
                    'top_3_accuracy': 0.95,
                    'ece': 0.02
                },
                {
                    'task': 'multi_label_component_routing',
                    'official_model': 'CatBoost MultiLabel',
                    'artifact_role': 'component_multilabel_official',
                    'feature_set': 'core_structured',
                    'holdout_rows': 120,
                    'macro_f1': 0.23,
                    'micro_f1': 0.46,
                    'recall_at_3': 0.68,
                    'precision_at_3': 0.30
                }
            ]
        )
        with (outputs_dir / 'component_textwave2b_calibration_manifest.json').open('w', encoding='utf-8') as handle:
            json.dump(
                {
                    'locked_holdout_baseline': {
                        'macro_f1': 0.70,
                        'top_1_accuracy': 0.80,
                        'top_3_accuracy': 0.92
                    },
                    'calibrated_holdout_metrics': {
                        'macro_f1': 0.75,
                        'top_1_accuracy': 0.85,
                        'top_3_accuracy': 0.95
                    }
                },
                handle
            )
        write_csv(
            outputs_dir / 'component_single_label_official_class_metrics.csv',
            [
                {'component_group': 'ENGINE', 'support': 20, 'precision': 0.9, 'recall': 0.8, 'f1': 0.85},
                {'component_group': 'BRAKES', 'support': 10, 'precision': 0.7, 'recall': 0.6, 'f1': 0.65}
            ]
        )
        write_csv(
            outputs_dir / 'component_single_label_official_confusion_major.csv',
            [
                {'true_group': 'ENGINE', 'pred_group': 'ENGINE', 'count': 18, 'row_share': 0.90},
                {'true_group': 'ENGINE', 'pred_group': 'BRAKES', 'count': 2, 'row_share': 0.10},
                {'true_group': 'BRAKES', 'pred_group': 'ENGINE', 'count': 3, 'row_share': 0.30},
                {'true_group': 'BRAKES', 'pred_group': 'BRAKES', 'count': 7, 'row_share': 0.70}
            ]
        )
        write_csv(
            outputs_dir / 'component_single_label_official_calibration.csv',
            [
                {
                    'calibration_method': 'power',
                    'calibration_alpha': 1.5,
                    'section': 'overall',
                    'bin': 'overall',
                    'count': 30,
                    'share': 1.0,
                    'accuracy': 0.85,
                    'avg_confidence': 0.87,
                    'gap': 0.02,
                    'ece': 0.02,
                    'multiclass_brier': 0.22
                },
                {
                    'calibration_method': 'power',
                    'calibration_alpha': 1.5,
                    'section': 'bin',
                    'bin': '(0.8, 0.9]',
                    'count': 30,
                    'share': 1.0,
                    'accuracy': 0.85,
                    'avg_confidence': 0.87,
                    'gap': 0.02,
                    'ece': '',
                    'multiclass_brier': ''
                }
            ]
        )
        write_csv(
            outputs_dir / 'component_multilabel_official_label_metrics.csv',
            [
                {'model': 'CatBoost MultiLabel', 'component_group': 'ENGINE', 'support': 20, 'precision': 0.4, 'recall': 0.8, 'f1': 0.53},
                {'model': 'CatBoost MultiLabel', 'component_group': 'BRAKES', 'support': 10, 'precision': 0.3, 'recall': 0.5, 'f1': 0.38}
            ]
        )
        result = generate_component_visuals(outputs_dir=outputs_dir, output_dir=figures_dir)

        assert result['index_path'].exists()
        assert len(result['figures']) == 6
        for row in result['figures']:
            assert (figures_dir / f"{row['figure']}.png").exists()
    finally:
        shutil.rmtree(test_root, ignore_errors=True)
