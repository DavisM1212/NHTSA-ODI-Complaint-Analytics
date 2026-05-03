import argparse
import json
from pathlib import Path

import pandas as pd

from src.config.contracts import (
    COMPONENT_MULTI_OFFICIAL_MANIFEST,
    COMPONENT_OFFICIAL_SUMMARY_CSV,
    COMPONENT_OFFICIAL_SUMMARY_JSON,
    COMPONENT_SINGLE_OFFICIAL_MANIFEST,
    NLP_EARLY_WARNING_OFFICIAL_MANIFEST,
    NLP_EARLY_WARNING_WATCHLIST_SUMMARY,
    README_END,
    README_START,
    SEVERITY_URGENCY_OFFICIAL_MANIFEST,
)
from src.config.paths import OUTPUTS_DIR, PROJECT_ROOT

ALLOWED_RELEASE_STATUSES = {'official', 'promoted'}


def load_manifest(manifest_path):
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")

    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def format_metric(value):
    if value is None:
        return 'n/a'
    try:
        return f'{float(value):.4f}'
    except Exception:
        return str(value)


def validate_release_status(manifest, manifest_name):
    status = str(manifest.get('promotion_status', '')).strip().lower()
    if status not in ALLOWED_RELEASE_STATUSES:
        allowed = ', '.join(sorted(ALLOWED_RELEASE_STATUSES))
        raise ValueError(f"{manifest_name} must record promotion_status as one of: {allowed}")
    if not bool(manifest.get('reporting_ready', False)):
        raise ValueError(f"{manifest_name} must set reporting_ready=true before publishing")


def require_dict(manifest, field_name, manifest_name):
    value = manifest.get(field_name)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{manifest_name} is missing required dict field: {field_name}")
    return value


def require_field(manifest, field_name, manifest_name):
    value = manifest.get(field_name)
    if value in {None, ''}:
        raise ValueError(f"{manifest_name} is missing required field: {field_name}")
    return value


def validate_single_manifest(single_manifest):
    manifest_name = COMPONENT_SINGLE_OFFICIAL_MANIFEST
    validate_release_status(single_manifest, manifest_name)
    require_field(single_manifest, 'artifact_role', manifest_name)
    require_field(single_manifest, 'official_model', manifest_name)
    require_field(single_manifest, 'structured_feature_set', manifest_name)
    require_field(single_manifest, 'calibration_method', manifest_name)
    require_field(single_manifest, 'selected_alpha', manifest_name)
    require_field(single_manifest, 'text_weight', manifest_name)
    require_dict(single_manifest, 'official_holdout_metrics', manifest_name)
    require_dict(single_manifest, 'artifacts', manifest_name)
    return single_manifest


def validate_multi_manifest(multi_manifest):
    manifest_name = COMPONENT_MULTI_OFFICIAL_MANIFEST
    validate_release_status(multi_manifest, manifest_name)
    require_field(multi_manifest, 'artifact_role', manifest_name)
    require_field(multi_manifest, 'selected_model', manifest_name)
    require_field(multi_manifest, 'selected_feature_set', manifest_name)
    require_field(multi_manifest, 'selected_threshold', manifest_name)
    require_field(multi_manifest, 'selected_iteration', manifest_name)
    require_dict(multi_manifest, 'official_holdout_metrics', manifest_name)
    require_dict(multi_manifest, 'artifacts', manifest_name)
    return multi_manifest


def validate_severity_manifest(severity_manifest):
    manifest_name = SEVERITY_URGENCY_OFFICIAL_MANIFEST
    status = str(severity_manifest.get('publish_status', '')).strip().lower()
    if status not in ALLOWED_RELEASE_STATUSES:
        allowed = ', '.join(sorted(ALLOWED_RELEASE_STATUSES))
        raise ValueError(f"{manifest_name} must record publish_status as one of: {allowed}")

    require_field(severity_manifest, 'scope', manifest_name)
    require_field(severity_manifest, 'target_col', manifest_name)
    require_field(severity_manifest, 'baseline_model_name', manifest_name)
    require_field(severity_manifest, 'official_model_name', manifest_name)
    locked_params = require_dict(severity_manifest, 'locked_params', manifest_name)
    validation_metrics = require_dict(severity_manifest, 'validation_metrics', manifest_name)
    holdout_metrics = require_dict(severity_manifest, 'holdout_metrics', manifest_name)

    if not isinstance(validation_metrics.get('official'), dict) or not validation_metrics['official']:
        raise ValueError(f"{manifest_name} is missing validation_metrics.official")
    if not isinstance(holdout_metrics.get('official'), dict) or not holdout_metrics['official']:
        raise ValueError(f"{manifest_name} is missing holdout_metrics.official")
    if 'text_weight' not in locked_params:
        raise ValueError(f"{manifest_name} is missing locked_params.text_weight")
    return severity_manifest


def validate_nlp_manifest(nlp_manifest):
    manifest_name = NLP_EARLY_WARNING_OFFICIAL_MANIFEST
    status = str(nlp_manifest.get('publish_status', '')).strip().lower()
    if status not in ALLOWED_RELEASE_STATUSES:
        allowed = ', '.join(sorted(ALLOWED_RELEASE_STATUSES))
        raise ValueError(f"{manifest_name} must record publish_status as one of: {allowed}")

    require_field(nlp_manifest, 'scope', manifest_name)
    topic_model = require_dict(nlp_manifest, 'topic_model', manifest_name)
    row_counts = require_dict(nlp_manifest, 'row_counts', manifest_name)
    require_field(topic_model, 'locked_topic_k', manifest_name)
    require_field(nlp_manifest, 'latest_watchlist_month', manifest_name)
    require_field(row_counts, 'watchlist_rows', manifest_name)
    require_field(row_counts, 'risk_monitor_rows', manifest_name)
    require_field(row_counts, 'recurring_large_signal_rows', manifest_name)
    return nlp_manifest


def load_watchlist_summary(summary_path):
    path = Path(summary_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing watchlist summary: {path}")

    summary = pd.read_csv(path)
    if 'month' not in summary.columns:
        raise ValueError(f"{path.name} must contain a month column")
    summary['month'] = pd.to_datetime(summary['month'])
    return summary


def build_severity_lines(severity_manifest):
    valid = severity_manifest['validation_metrics']['official']
    holdout = severity_manifest['holdout_metrics']['official']
    locked = severity_manifest['locked_params']
    return [
        '#### Severity urgency benchmark',
        '',
        '- Scope: official complaint-level severity urgency benchmark',
        f"- Target: `{severity_manifest.get('target_col', 'n/a')}`",
        f"- Baseline: `{severity_manifest.get('baseline_model_name', 'n/a')}`",
        f"- Model: `{severity_manifest.get('official_model_name', 'n/a')}`",
        f"- Text weight: `{locked.get('text_weight', 'n/a')}`",
        f"- Validation PR-AUC / Brier: `{format_metric(valid.get('pr_auc'))}` / `{format_metric(valid.get('brier_score'))}`",
        f"- Validation top-5% recall / precision: `{format_metric(valid.get('recall_top_5pct'))}` / `{format_metric(valid.get('precision_top_5pct'))}`",
        f"- Holdout PR-AUC / Brier: `{format_metric(holdout.get('pr_auc'))}` / `{format_metric(holdout.get('brier_score'))}`",
        f"- Holdout top-5% recall / precision: `{format_metric(holdout.get('recall_top_5pct'))}` / `{format_metric(holdout.get('precision_top_5pct'))}`",
        ''
    ]


def build_single_lines(single_manifest):
    holdout = single_manifest['official_holdout_metrics']
    return [
        '#### Single-label component benchmark',
        '',
        '- Scope: official single-label component complaint benchmark',
        f"- Model: `{single_manifest.get('official_model', single_manifest.get('family_name', 'n/a'))}`",
        f"- Inputs: complaint narrative text + `{single_manifest.get('structured_feature_set', 'n/a')}` structured companion features",
        f"- Text weight: `{single_manifest.get('text_weight', 'n/a')}`",
        f"- Final text model: `{single_manifest.get('final_linear_model', 'n/a')}`",
        f"- Calibration: `{single_manifest.get('calibration_method', 'n/a')}` alpha `{single_manifest.get('selected_alpha', 'n/a')}` from `{single_manifest.get('calibration_source', 'n/a')}`",
        f"- Structured branch iteration: `{single_manifest.get('selected_iteration', holdout.get('selected_iteration', 'n/a'))}`",
        f"- Holdout macro F1: `{format_metric(holdout.get('macro_f1'))}`",
        f"- Holdout top-1 accuracy: `{format_metric(holdout.get('top_1_accuracy'))}`",
        f"- Holdout top-3 accuracy: `{format_metric(holdout.get('top_3_accuracy'))}`",
        f"- Holdout calibration ECE: `{format_metric(single_manifest.get('holdout_ece'))}`",
        f"- Release status: `{single_manifest.get('promotion_status', 'n/a')}`",
        ''
    ]


def build_multi_lines(multi_manifest):
    holdout = multi_manifest['official_holdout_metrics']
    return [
        '#### Multi-label routing benchmark',
        '',
        '- Scope: official multi-label complaint routing benchmark',
        f"- Model: `{multi_manifest.get('selected_model', 'n/a')}`",
        f"- Feature set: `{multi_manifest.get('selected_feature_set', 'n/a')}`",
        f"- Threshold: `{multi_manifest.get('selected_threshold', 'n/a')}`",
        f"- Selected iteration: `{multi_manifest.get('selected_iteration', 'n/a')}`",
        f"- Holdout macro F1: `{format_metric(holdout.get('macro_f1'))}`",
        f"- Holdout micro F1: `{format_metric(holdout.get('micro_f1'))}`",
        f"- Holdout recall@3: `{format_metric(holdout.get('recall_at_3'))}`",
        f"- Holdout precision@3: `{format_metric(holdout.get('precision_at_3'))}`",
        f"- Release status: `{multi_manifest.get('promotion_status', 'n/a')}`",
        ''
    ]


def build_nlp_lines(nlp_manifest, watchlist_summary):
    topic_model = nlp_manifest['topic_model']
    row_counts = nlp_manifest['row_counts']
    latest_month = pd.to_datetime(nlp_manifest['latest_watchlist_month'])

    lines = [
        '#### NLP early-warning snapshot',
        '',
        f"- Scope: `{nlp_manifest.get('scope', 'n/a')}`",
        f"- Locked topic count: `{topic_model.get('locked_topic_k', 'n/a')}`",
        f"- Development window end: `{pd.to_datetime(nlp_manifest['time_windows']['development_end']).strftime('%Y-%m')}`",
        f"- Forward watchlist window start: `{pd.to_datetime(nlp_manifest['time_windows']['forward_start']).strftime('%Y-%m')}`",
        f"- Latest watchlist month: `{latest_month.strftime('%Y-%m')}`",
        f"- Watchlist rows: `{int(row_counts.get('watchlist_rows', 0))}`",
        f"- Risk monitor rows: `{int(row_counts.get('risk_monitor_rows', 0))}`",
        f"- Recurring large-signal rows: `{int(row_counts.get('recurring_large_signal_rows', 0))}`",
        ''
    ]

    latest = watchlist_summary.loc[watchlist_summary['month'] == latest_month].copy()
    if latest.empty:
        lines.extend(
            [
                '- Latest-month signal examples: `n/a`',
                ''
            ]
        )
        return lines

    latest = latest.sort_values(
        ['max_component_watchlist_score', 'complaints'],
        ascending=[False, False]
    ).drop_duplicates(subset=['topic_id']).head(3)

    lines.append('Latest-month signal examples:')
    for _, row in latest.iterrows():
        cohort = f"{row['maketxt']} {row['modeltxt']} {int(row['yeartxt'])}"
        lines.append(
            f"- `{cohort}` | `{row['topic_label']}` | `{int(row['complaints'])}` complaints | `{row['best_signal_tier']}`"
        )
    lines.append('')
    return lines


def build_summary_rows(single_manifest=None, multi_manifest=None):
    rows = []

    if single_manifest is not None:
        holdout = single_manifest['official_holdout_metrics']
        rows.append(
            {
                'task': 'single_label_component',
                'official_model': single_manifest.get('official_model', single_manifest.get('family_name', 'n/a')),
                'artifact_role': single_manifest.get('artifact_role', 'component_single_label_official'),
                'feature_set': single_manifest.get('structured_feature_set', 'n/a'),
                'holdout_rows': holdout.get('rows'),
                'macro_f1': holdout.get('macro_f1'),
                'top_1_accuracy': holdout.get('top_1_accuracy'),
                'top_3_accuracy': holdout.get('top_3_accuracy'),
                'ece': single_manifest.get('holdout_ece'),
                'calibration_method': single_manifest.get('calibration_method'),
                'calibration_alpha': single_manifest.get('selected_alpha'),
                'status': single_manifest.get('promotion_status', 'official')
            }
        )

    if multi_manifest is not None:
        holdout = multi_manifest['official_holdout_metrics']
        rows.append(
            {
                'task': 'multi_label_component_routing',
                'official_model': multi_manifest.get('selected_model', holdout.get('model', 'n/a')),
                'artifact_role': multi_manifest.get('artifact_role', 'component_multilabel_official'),
                'feature_set': multi_manifest.get('selected_feature_set', 'n/a'),
                'holdout_rows': holdout.get('rows'),
                'macro_f1': holdout.get('macro_f1'),
                'micro_f1': holdout.get('micro_f1'),
                'recall_at_3': holdout.get('recall_at_3'),
                'precision_at_3': holdout.get('precision_at_3'),
                'label_coverage': holdout.get('label_coverage'),
                'threshold': holdout.get('threshold', multi_manifest.get('selected_threshold')),
                'selected_iteration': holdout.get('selected_iteration', multi_manifest.get('selected_iteration')),
                'status': multi_manifest.get('promotion_status', 'official')
            }
        )

    return rows


def write_summary_artifacts(single_manifest=None, multi_manifest=None, summary_csv_path=None, summary_json_path=None):
    rows = build_summary_rows(single_manifest=single_manifest, multi_manifest=multi_manifest)
    csv_path = Path(summary_csv_path) if summary_csv_path is not None else OUTPUTS_DIR / COMPONENT_OFFICIAL_SUMMARY_CSV
    json_path = Path(summary_json_path) if summary_json_path is not None else OUTPUTS_DIR / COMPONENT_OFFICIAL_SUMMARY_JSON

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        import pandas as pd

        pd.DataFrame(rows).to_csv(csv_path, index=False)
    else:
        csv_path.write_text('', encoding='utf-8')

    with json_path.open('w', encoding='utf-8') as handle:
        json.dump({'benchmarks': rows}, handle, indent=2)

    return csv_path, json_path


def build_readme_block(severity_manifest=None, single_manifest=None, multi_manifest=None, nlp_manifest=None, nlp_watchlist_summary=None):
    lines = [
        README_START,
        '### Generated Benchmark Snapshot',
        '',
        'This section is generated from the official severity, component, and NLP early-warning artifacts in `data/outputs/`.',
        'Severity reports the locked primary-target urgency rule on `valid_2025` plus the `2026` reference check.',
        'The published component-model scores come from the untouched `2026` holdout.',
        ''
    ]
    if severity_manifest is not None:
        lines.extend(build_severity_lines(severity_manifest))
    lines.extend(build_single_lines(single_manifest))
    lines.extend(build_multi_lines(multi_manifest))
    if nlp_manifest is not None and nlp_watchlist_summary is not None:
        lines.extend(build_nlp_lines(nlp_manifest, nlp_watchlist_summary))
    lines.append(README_END)
    return '\n'.join(lines)


def update_component_readme(
    single_manifest_path=None,
    multi_manifest_path=None,
    severity_manifest_path=None,
    nlp_manifest_path=None,
    nlp_watchlist_summary_path=None,
    readme_path=None,
    write_summary=True
):
    readme_path = Path(readme_path) if readme_path is not None else PROJECT_ROOT / 'README.md'
    text = readme_path.read_text(encoding='utf-8')

    single_path = Path(single_manifest_path) if single_manifest_path is not None else OUTPUTS_DIR / COMPONENT_SINGLE_OFFICIAL_MANIFEST
    multi_path = Path(multi_manifest_path) if multi_manifest_path is not None else OUTPUTS_DIR / COMPONENT_MULTI_OFFICIAL_MANIFEST

    single_manifest = validate_single_manifest(load_manifest(single_path))
    multi_manifest = validate_multi_manifest(load_manifest(multi_path))
    severity_manifest = None
    if severity_manifest_path is not None:
        severity_manifest = validate_severity_manifest(load_manifest(severity_manifest_path))
    nlp_manifest = None
    nlp_watchlist_summary = None
    if nlp_manifest_path is not None:
        nlp_manifest = validate_nlp_manifest(load_manifest(nlp_manifest_path))
    if nlp_watchlist_summary_path is not None:
        nlp_watchlist_summary = load_watchlist_summary(nlp_watchlist_summary_path)
    block = build_readme_block(
        severity_manifest=severity_manifest,
        single_manifest=single_manifest,
        multi_manifest=multi_manifest,
        nlp_manifest=nlp_manifest,
        nlp_watchlist_summary=nlp_watchlist_summary
    )

    if README_START not in text or README_END not in text:
        raise ValueError("README benchmark markers not found")

    start_idx = text.index(README_START)
    end_idx = text.index(README_END) + len(README_END)
    updated = text[:start_idx] + block + text[end_idx:]
    readme_path.write_text(updated, encoding='utf-8')
    if write_summary:
        write_summary_artifacts(single_manifest=single_manifest, multi_manifest=multi_manifest)
    return readme_path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Update the README benchmark section from the official severity and component artifacts'
    )
    parser.add_argument('--single-manifest', default=None)
    parser.add_argument('--multi-manifest', default=None)
    parser.add_argument('--severity-manifest', default=str(OUTPUTS_DIR / SEVERITY_URGENCY_OFFICIAL_MANIFEST))
    parser.add_argument('--nlp-manifest', default=str(OUTPUTS_DIR / NLP_EARLY_WARNING_OFFICIAL_MANIFEST))
    parser.add_argument('--nlp-watchlist-summary', default=str(OUTPUTS_DIR / NLP_EARLY_WARNING_WATCHLIST_SUMMARY))
    parser.add_argument('--readme-path', default=None)
    parser.add_argument('--no-summary', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    readme_path = update_component_readme(
        single_manifest_path=args.single_manifest,
        multi_manifest_path=args.multi_manifest,
        severity_manifest_path=args.severity_manifest,
        nlp_manifest_path=args.nlp_manifest,
        nlp_watchlist_summary_path=args.nlp_watchlist_summary,
        readme_path=args.readme_path,
        write_summary=not args.no_summary
    )
    print(f'[write] {readme_path}')
    print('[done] README benchmark section refreshed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
