import json
import shutil

import pandas as pd

from src.config.contracts import (
    SEVERITY_URGENCY_OFFICIAL_CALIBRATION,
    SEVERITY_URGENCY_OFFICIAL_MANIFEST,
    SEVERITY_URGENCY_OFFICIAL_METRICS,
    SEVERITY_URGENCY_OFFICIAL_REVIEW_BUDGETS,
)
from src.config.paths import OUTPUTS_DIR
from src.reporting.severity_visuals import generate_severity_visuals


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def build_manifest():
    return {
        'scope': 'official severity urgency benchmark',
        'publish_status': 'official',
        'target_col': 'severity_primary_flag',
        'baseline_model_name': 'dummy_prior',
        'official_model_name': 'late_fusion_sigmoid',
        'split_summary': [
            {'split': 'train', 'rows': 1000, 'positive_rate': 0.06},
            {'split': 'valid_2025', 'rows': 250, 'positive_rate': 0.05},
            {'split': 'holdout_2026', 'rows': 200, 'positive_rate': 0.06}
        ]
    }


def build_metrics_rows():
    rows = []
    for split_name in ['valid_2025', 'holdout_2026']:
        rows.extend(
            [
                {
                    'model': 'dummy_prior',
                    'split': split_name,
                    'pr_auc': 0.05,
                    'roc_auc': 0.50,
                    'f1': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'brier_score': 0.05,
                    'recall_top_5pct': 0.04,
                    'precision_top_5pct': 0.05,
                    'recall_top_10pct': 0.08,
                    'is_baseline': True,
                    'is_official': False
                },
                {
                    'model': 'late_fusion_raw',
                    'split': split_name,
                    'pr_auc': 0.82,
                    'roc_auc': 0.96,
                    'f1': 0.69,
                    'precision': 0.70,
                    'recall': 0.68,
                    'brier_score': 0.038,
                    'recall_top_5pct': 0.75,
                    'precision_top_5pct': 0.79,
                    'recall_top_10pct': 0.88,
                    'is_baseline': False,
                    'is_official': False
                },
                {
                    'model': 'late_fusion_sigmoid',
                    'split': split_name,
                    'pr_auc': 0.83,
                    'roc_auc': 0.96,
                    'f1': 0.77,
                    'precision': 0.81,
                    'recall': 0.74,
                    'brier_score': 0.018,
                    'recall_top_5pct': 0.756,
                    'precision_top_5pct': 0.795,
                    'recall_top_10pct': 0.884,
                    'is_baseline': False,
                    'is_official': True
                },
                {
                    'model': 'late_fusion_isotonic',
                    'split': split_name,
                    'pr_auc': 0.821,
                    'roc_auc': 0.96,
                    'f1': 0.77,
                    'precision': 0.81,
                    'recall': 0.74,
                    'brier_score': 0.017,
                    'recall_top_5pct': 0.756,
                    'precision_top_5pct': 0.795,
                    'recall_top_10pct': 0.884,
                    'is_baseline': False,
                    'is_official': False
                }
            ]
        )
    return rows


def build_review_budget_rows():
    rows = []
    budgets = [
        (0.01, 'top_1pct', 10, 0.01),
        (0.02, 'top_2pct', 20, 0.02),
        (0.05, 'top_5pct', 50, 0.05),
        (0.10, 'top_10pct', 100, 0.10)
    ]
    for split_name in ['valid_2025', 'holdout_2026']:
        for model_name, is_baseline, is_official, precision_base, recall_base in [
            ('dummy_prior', True, False, 0.05, 0.04),
            ('late_fusion_sigmoid', False, True, 0.80, 0.75)
        ]:
            for fraction, label, flagged_rows, flagged_share in budgets:
                rows.append(
                    {
                        'model': model_name,
                        'split': split_name,
                        'budget_fraction': fraction,
                        'budget_label': label,
                        'flagged_rows': flagged_rows,
                        'flagged_share': flagged_share,
                        'severe_cases_captured': int(round(flagged_rows * precision_base)),
                        'recall_within_flagged_set': recall_base + (0.10 if fraction == 0.10 and model_name == 'late_fusion_sigmoid' else 0.0),
                        'precision_within_flagged_set': precision_base - (0.25 if fraction == 0.10 and model_name == 'late_fusion_sigmoid' else 0.0),
                        'lift_vs_base_rate': 15.0 if model_name == 'late_fusion_sigmoid' else 1.0,
                        'score_cutoff': 0.90 - fraction,
                        'is_baseline': is_baseline,
                        'is_official': is_official
                    }
                )
    return rows


def build_calibration_rows():
    rows = []
    valid_curves = {
        'late_fusion_raw': [(0.02, 0.01), (0.08, 0.04), (0.18, 0.10), (0.70, 0.47)],
        'late_fusion_sigmoid': [(0.01, 0.01), (0.04, 0.04), (0.11, 0.10), (0.48, 0.47)],
        'late_fusion_isotonic': [(0.01, 0.01), (0.04, 0.04), (0.10, 0.10), (0.47, 0.47)]
    }
    holdout_curves = {
        'late_fusion_raw': [(0.02, 0.00), (0.10, 0.04), (0.20, 0.12), (0.77, 0.54)],
        'late_fusion_sigmoid': [(0.00, 0.00), (0.04, 0.04), (0.12, 0.11), (0.60, 0.54)]
    }

    for model_name, points in valid_curves.items():
        for idx, (avg_score, observed_rate) in enumerate(points):
            rows.append(
                {
                    'model': model_name,
                    'split': 'valid_2025',
                    'bin': idx,
                    'rows': 50,
                    'avg_score': avg_score,
                    'observed_rate': observed_rate,
                    'score_gap': avg_score - observed_rate,
                    'is_official': model_name == 'late_fusion_sigmoid'
                }
            )

    for model_name, points in holdout_curves.items():
        for idx, (avg_score, observed_rate) in enumerate(points):
            rows.append(
                {
                    'model': model_name,
                    'split': 'holdout_2026',
                    'bin': idx,
                    'rows': 40,
                    'avg_score': avg_score,
                    'observed_rate': observed_rate,
                    'score_gap': avg_score - observed_rate,
                    'is_official': model_name == 'late_fusion_sigmoid'
                }
            )
    return rows


def test_generate_severity_visuals_writes_expected_files():
    test_root = OUTPUTS_DIR / '_severity_visuals_test'
    outputs_dir = test_root / 'outputs'
    figures_dir = test_root / 'figures'
    shutil.rmtree(test_root, ignore_errors=True)
    try:
        outputs_dir.mkdir(parents=True)

        with (outputs_dir / SEVERITY_URGENCY_OFFICIAL_MANIFEST).open('w', encoding='utf-8') as handle:
            json.dump(build_manifest(), handle)

        write_csv(outputs_dir / SEVERITY_URGENCY_OFFICIAL_METRICS, build_metrics_rows())
        write_csv(outputs_dir / SEVERITY_URGENCY_OFFICIAL_REVIEW_BUDGETS, build_review_budget_rows())
        write_csv(outputs_dir / SEVERITY_URGENCY_OFFICIAL_CALIBRATION, build_calibration_rows())

        result = generate_severity_visuals(outputs_dir=outputs_dir, output_dir=figures_dir)

        assert result['index_path'].exists()
        assert len(result['figures']) == 6
        for row in result['figures']:
            assert (figures_dir / f"{row['figure']}.png").exists()
    finally:
        shutil.rmtree(test_root, ignore_errors=True)
