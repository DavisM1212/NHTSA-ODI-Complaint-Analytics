import json
import shutil

import pandas as pd

from src.config.paths import OUTPUTS_DIR
from src.reporting.watchlist_visuals import generate_watchlist_visuals


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def build_manifest():
    return {
        'artifact_role': 'nlp_early_warning_official',
        'scope': 'official lemma-based NLP early-warning pipeline',
        'time_windows': {
            'development_end': '2024-12-31 00:00:00',
            'forward_start': '2025-01-01 00:00:00'
        },
        'topic_model': {
            'locked_topic_k': 20,
            'recommended_topic_k': 20
        }
    }


def build_topic_library_rows():
    return [
        {
            'topic_id': 1,
            'topic_label': 'Electric power steering assist loss',
            'topic_category': 'defect',
            'watchlist_group': 'defect_watchlist',
            'topic_top_terms': 'power | steering | assist',
            'development_share': 0.040,
            'forward_share': 0.052,
            'median_topic_weight': 0.43,
            'share_percent_change': 30.0
        },
        {
            'topic_id': 6,
            'topic_label': 'Transmission shifting / gear engagement issue',
            'topic_category': 'defect',
            'watchlist_group': 'defect_watchlist',
            'topic_top_terms': 'transmission | gear | shifting',
            'development_share': 0.059,
            'forward_share': 0.077,
            'median_topic_weight': 0.52,
            'share_percent_change': 31.7
        },
        {
            'topic_id': 10,
            'topic_label': 'Steering wheel binding / difficult turning issue',
            'topic_category': 'defect',
            'watchlist_group': 'defect_watchlist',
            'topic_top_terms': 'steering | wheel | turning',
            'development_share': 0.036,
            'forward_share': 0.049,
            'median_topic_weight': 0.57,
            'share_percent_change': 36.1
        },
        {
            'topic_id': 15,
            'topic_label': 'Oil consumption / low oil issue',
            'topic_category': 'defect',
            'watchlist_group': 'defect_watchlist',
            'topic_top_terms': 'oil | consumption | low',
            'development_share': 0.048,
            'forward_share': 0.056,
            'median_topic_weight': 0.45,
            'share_percent_change': 15.7
        },
        {
            'topic_id': 5,
            'topic_label': 'Recall notice / VIN eligibility / remedy process',
            'topic_category': 'recall_process',
            'watchlist_group': 'process',
            'topic_top_terms': 'recall | vin | remedy',
            'development_share': 0.057,
            'forward_share': 0.086,
            'median_topic_weight': 0.39,
            'share_percent_change': 50.4
        }
    ]


def build_watchlist_rows():
    return [
        {
            'month': '2026-01-01',
            'mfr_name': 'FORD MOTOR COMPANY',
            'maketxt': 'FORD',
            'modeltxt': 'F-150',
            'yeartxt': '2017',
            'component_group': 'POWER TRAIN',
            'topic_id': 6,
            'topic_label': 'Transmission shifting / gear engagement issue',
            'topic_category': 'defect',
            'complaints': 18,
            'unique_states': 12,
            'rolling_6mo_avg': 4.0,
            'growth_vs_6mo_avg': 4.5,
            'growth_delta_6mo': 14.0,
            'severity_broad_rate': 0.0,
            'critical_event_near_operation_rate': 0.05,
            'in_operation_rate': 0.50,
            'highway_rate': 0.20,
            'high_speed_rate': 0.26,
            'avg_topic_strength': 0.51,
            'watchlist_score': 18.2,
            'watchlist_reason': '4.5x 6-mo baseline; strong topic match',
            'signal_tier': 'High-confidence signal',
            'month_rank': 1
        },
        {
            'month': '2026-02-01',
            'mfr_name': 'FORD MOTOR COMPANY',
            'maketxt': 'FORD',
            'modeltxt': 'F-150',
            'yeartxt': '2017',
            'component_group': 'POWER TRAIN',
            'topic_id': 6,
            'topic_label': 'Transmission shifting / gear engagement issue',
            'topic_category': 'defect',
            'complaints': 38,
            'unique_states': 21,
            'rolling_6mo_avg': 6.0,
            'growth_vs_6mo_avg': 6.3,
            'growth_delta_6mo': 32.0,
            'severity_broad_rate': 0.02,
            'critical_event_near_operation_rate': 0.05,
            'in_operation_rate': 0.50,
            'highway_rate': 0.18,
            'high_speed_rate': 0.26,
            'avg_topic_strength': 0.51,
            'watchlist_score': 24.1,
            'watchlist_reason': '6.3x 6-mo baseline; strong topic match',
            'signal_tier': 'High-confidence signal',
            'month_rank': 1
        },
        {
            'month': '2026-02-01',
            'mfr_name': 'MAZDA MOTOR CORP.',
            'maketxt': 'MAZDA',
            'modeltxt': 'CX-90',
            'yeartxt': '2024',
            'component_group': 'STEERING',
            'topic_id': 10,
            'topic_label': 'Steering wheel binding / difficult turning issue',
            'topic_category': 'defect',
            'complaints': 18,
            'unique_states': 14,
            'rolling_6mo_avg': 3.2,
            'growth_vs_6mo_avg': 5.7,
            'growth_delta_6mo': 14.8,
            'severity_broad_rate': 0.0,
            'critical_event_near_operation_rate': 0.0,
            'in_operation_rate': 0.17,
            'highway_rate': 0.17,
            'high_speed_rate': 0.17,
            'avg_topic_strength': 0.57,
            'watchlist_score': 17.8,
            'watchlist_reason': '5.7x 6-mo baseline; strong topic match',
            'signal_tier': 'High-confidence signal',
            'month_rank': 2
        },
        {
            'month': '2026-02-01',
            'mfr_name': 'HYUNDAI MOTOR AMERICA',
            'maketxt': 'HYUNDAI',
            'modeltxt': 'IONIQ 5',
            'yeartxt': '2024',
            'component_group': 'ELECTRICAL SYSTEM',
            'topic_id': 1,
            'topic_label': 'Electric power steering assist loss',
            'topic_category': 'defect',
            'complaints': 7,
            'unique_states': 6,
            'rolling_6mo_avg': 0.5,
            'growth_vs_6mo_avg': 7.0,
            'growth_delta_6mo': 6.5,
            'severity_broad_rate': 0.0,
            'critical_event_near_operation_rate': 0.71,
            'in_operation_rate': 0.71,
            'highway_rate': 0.43,
            'high_speed_rate': 0.43,
            'avg_topic_strength': 0.36,
            'watchlist_score': 17.0,
            'watchlist_reason': '7.0x 6-mo baseline; critical event while in operation',
            'signal_tier': 'Moderate signal',
            'month_rank': 3
        },
        {
            'month': '2026-02-01',
            'mfr_name': 'CHEVROLET',
            'maketxt': 'CHEVROLET',
            'modeltxt': 'EQUINOX',
            'yeartxt': '2019',
            'component_group': 'ENGINE',
            'topic_id': 15,
            'topic_label': 'Oil consumption / low oil issue',
            'topic_category': 'defect',
            'complaints': 4,
            'unique_states': 4,
            'rolling_6mo_avg': 1.5,
            'growth_vs_6mo_avg': 2.7,
            'growth_delta_6mo': 2.5,
            'severity_broad_rate': 0.0,
            'critical_event_near_operation_rate': 0.0,
            'in_operation_rate': 0.0,
            'highway_rate': 0.0,
            'high_speed_rate': 0.0,
            'avg_topic_strength': 0.41,
            'watchlist_score': 8.8,
            'watchlist_reason': '2.7x 6-mo baseline; strong topic match',
            'signal_tier': 'Early signal',
            'month_rank': 4
        }
    ]


def build_watchlist_summary_rows():
    return [
        {
            'month': '2026-01-01',
            'mfr_name': 'FORD MOTOR COMPANY',
            'maketxt': 'FORD',
            'modeltxt': 'F-150',
            'yeartxt': '2017',
            'topic_id': 6,
            'topic_label': 'Transmission shifting / gear engagement issue',
            'component_groups': 'POWER TRAIN',
            'complaints': 18,
            'unique_states': 12,
            'max_component_watchlist_score': 18.2,
            'avg_topic_strength': 0.51,
            'severity_broad_rate': 0.0,
            'critical_event_near_operation_rate': 0.05,
            'highway_rate': 0.20,
            'high_speed_rate': 0.26,
            'best_signal_tier': 'High-confidence signal'
        },
        {
            'month': '2026-02-01',
            'mfr_name': 'FORD MOTOR COMPANY',
            'maketxt': 'FORD',
            'modeltxt': 'F-150',
            'yeartxt': '2017',
            'topic_id': 6,
            'topic_label': 'Transmission shifting / gear engagement issue',
            'component_groups': 'ELECTRICAL SYSTEM | POWER TRAIN',
            'complaints': 39,
            'unique_states': 21,
            'max_component_watchlist_score': 24.1,
            'avg_topic_strength': 0.51,
            'severity_broad_rate': 0.02,
            'critical_event_near_operation_rate': 0.05,
            'highway_rate': 0.18,
            'high_speed_rate': 0.28,
            'best_signal_tier': 'High-confidence signal'
        },
        {
            'month': '2026-02-01',
            'mfr_name': 'MAZDA MOTOR CORP.',
            'maketxt': 'MAZDA',
            'modeltxt': 'CX-90',
            'yeartxt': '2024',
            'topic_id': 10,
            'topic_label': 'Steering wheel binding / difficult turning issue',
            'component_groups': 'STEERING',
            'complaints': 18,
            'unique_states': 14,
            'max_component_watchlist_score': 17.8,
            'avg_topic_strength': 0.57,
            'severity_broad_rate': 0.0,
            'critical_event_near_operation_rate': 0.0,
            'highway_rate': 0.17,
            'high_speed_rate': 0.17,
            'best_signal_tier': 'High-confidence signal'
        },
        {
            'month': '2026-02-01',
            'mfr_name': 'HYUNDAI MOTOR AMERICA',
            'maketxt': 'HYUNDAI',
            'modeltxt': 'IONIQ 5',
            'yeartxt': '2024',
            'topic_id': 1,
            'topic_label': 'Electric power steering assist loss',
            'component_groups': 'ELECTRICAL SYSTEM | POWER TRAIN',
            'complaints': 7,
            'unique_states': 6,
            'max_component_watchlist_score': 17.0,
            'avg_topic_strength': 0.36,
            'severity_broad_rate': 0.0,
            'critical_event_near_operation_rate': 0.71,
            'highway_rate': 0.43,
            'high_speed_rate': 0.43,
            'best_signal_tier': 'Moderate signal'
        },
        {
            'month': '2026-02-01',
            'mfr_name': 'CHEVROLET',
            'maketxt': 'CHEVROLET',
            'modeltxt': 'EQUINOX',
            'yeartxt': '2019',
            'topic_id': 15,
            'topic_label': 'Oil consumption / low oil issue',
            'component_groups': 'ENGINE',
            'complaints': 4,
            'unique_states': 4,
            'max_component_watchlist_score': 8.8,
            'avg_topic_strength': 0.41,
            'severity_broad_rate': 0.0,
            'critical_event_near_operation_rate': 0.0,
            'highway_rate': 0.0,
            'high_speed_rate': 0.0,
            'best_signal_tier': 'Early signal'
        }
    ]


def build_risk_rows():
    return [
        {
            'month': '2025-12-01',
            'mfr_name': 'FORD MOTOR COMPANY',
            'maketxt': 'FORD',
            'modeltxt': 'F-150',
            'yeartxt': '2018',
            'component_group': 'POWER TRAIN',
            'topic_id': 6,
            'topic_label': 'Transmission shifting / gear engagement issue',
            'topic_category': 'defect',
            'complaints': 5,
            'unique_states': 4,
            'rolling_6mo_avg': 3.0,
            'growth_vs_6mo_avg': 1.67,
            'growth_delta_6mo': 2.0,
            'severity_broad_rate': 0.0,
            'critical_event_near_operation_rate': 0.0,
            'high_speed_rate': 0.40,
            'avg_topic_strength': 0.44,
            'watchlist_score': 6.1,
            'monitor_reason': 'high-speed context',
            'month_rank': 1
        },
        {
            'month': '2026-02-01',
            'mfr_name': 'FORD MOTOR COMPANY',
            'maketxt': 'FORD',
            'modeltxt': 'F-150',
            'yeartxt': '2018',
            'component_group': 'POWER TRAIN',
            'topic_id': 6,
            'topic_label': 'Transmission shifting / gear engagement issue',
            'topic_category': 'defect',
            'complaints': 4,
            'unique_states': 4,
            'rolling_6mo_avg': 3.0,
            'growth_vs_6mo_avg': 1.33,
            'growth_delta_6mo': 1.0,
            'severity_broad_rate': 0.0,
            'critical_event_near_operation_rate': 0.0,
            'high_speed_rate': 0.75,
            'avg_topic_strength': 0.35,
            'watchlist_score': 3.7,
            'monitor_reason': 'high-speed context',
            'month_rank': 1
        }
    ]


def build_recurring_rows():
    return [
        {
            'maketxt': 'ACURA',
            'modeltxt': 'RDX',
            'yeartxt': '2015',
            'component_group': 'EXTERIOR LIGHTING',
            'topic_id': 18,
            'topic_label': 'Headlight / low beam visibility issue',
            'mfr_name': 'HONDA',
            'months_flagged': 16,
            'first_month': '2020-12-01',
            'latest_month': '2025-10-01',
            'max_complaints': 21,
            'avg_complaints': 8.19,
            'median_complaints': 6.0,
            'max_growth_vs_6mo_avg': 4.0,
            'avg_topic_strength': 0.85,
            'max_watchlist_score': 15.09,
            'high_confidence_months': 6,
            'moderate_or_higher_months': 13,
            'best_signal_tier': 'High-confidence signal'
        },
        {
            'maketxt': 'HONDA',
            'modeltxt': 'CIVIC',
            'yeartxt': '2016',
            'component_group': 'STEERING',
            'topic_id': 10,
            'topic_label': 'Steering wheel binding / difficult turning issue',
            'mfr_name': 'HONDA',
            'months_flagged': 16,
            'first_month': '2020-02-01',
            'latest_month': '2025-06-01',
            'max_complaints': 15,
            'avg_complaints': 8.0,
            'median_complaints': 8.0,
            'max_growth_vs_6mo_avg': 6.86,
            'avg_topic_strength': 0.54,
            'max_watchlist_score': 20.97,
            'high_confidence_months': 4,
            'moderate_or_higher_months': 13,
            'best_signal_tier': 'High-confidence signal'
        },
        {
            'maketxt': 'CHEVROLET',
            'modeltxt': 'MALIBU',
            'yeartxt': '2009',
            'component_group': 'STEERING',
            'topic_id': 1,
            'topic_label': 'Electric power steering assist loss',
            'mfr_name': 'GENERAL MOTORS',
            'months_flagged': 16,
            'first_month': '2020-01-01',
            'latest_month': '2025-02-01',
            'max_complaints': 12,
            'avg_complaints': 5.5,
            'median_complaints': 4.5,
            'max_growth_vs_6mo_avg': 5.0,
            'avg_topic_strength': 0.48,
            'max_watchlist_score': 13.24,
            'high_confidence_months': 0,
            'moderate_or_higher_months': 8,
            'best_signal_tier': 'Moderate signal'
        }
    ]


def test_generate_watchlist_visuals_writes_expected_files():
    test_root = OUTPUTS_DIR / '_watchlist_visuals_test'
    outputs_dir = test_root / 'outputs'
    figures_dir = test_root / 'figures'
    shutil.rmtree(test_root, ignore_errors=True)
    try:
        outputs_dir.mkdir(parents=True)

        with (outputs_dir / 'nlp_early_warning_official_manifest.json').open('w', encoding='utf-8') as handle:
            json.dump(build_manifest(), handle)

        write_csv(outputs_dir / 'nlp_early_warning_topic_library.csv', build_topic_library_rows())
        write_csv(outputs_dir / 'nlp_early_warning_watchlist.csv', build_watchlist_rows())
        write_csv(outputs_dir / 'nlp_early_warning_watchlist_summary.csv', build_watchlist_summary_rows())
        write_csv(outputs_dir / 'nlp_early_warning_risk_monitor.csv', build_risk_rows())
        write_csv(outputs_dir / 'nlp_early_warning_recurring_large_signals.csv', build_recurring_rows())

        result = generate_watchlist_visuals(outputs_dir=outputs_dir, output_dir=figures_dir)

        assert result['index_path'].exists()
        assert len(result['figures']) == 5
        for row in result['figures']:
            assert (figures_dir / f"{row['figure']}.png").exists()
        assert (figures_dir / 'nlp_early_warning_figure_index.csv').exists()
    finally:
        shutil.rmtree(test_root, ignore_errors=True)
