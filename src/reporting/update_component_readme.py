import argparse
import json
from pathlib import Path

from src.config.contracts import (
    COMPONENT_MULTI_OFFICIAL_MANIFEST,
    COMPONENT_OFFICIAL_SUMMARY_CSV,
    COMPONENT_OFFICIAL_SUMMARY_JSON,
    COMPONENT_SINGLE_OFFICIAL_MANIFEST,
    README_END,
    README_START,
)
from src.config.paths import OUTPUTS_DIR, PROJECT_ROOT

ALLOWED_RELEASE_STATUSES = {'official', 'promoted'}


def load_manifest(manifest_path):
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f'Missing manifest: {path}')

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
        raise ValueError(f'{manifest_name} must record promotion_status as one of: {allowed}')
    if not bool(manifest.get('reporting_ready', False)):
        raise ValueError(f'{manifest_name} must set reporting_ready=true before publishing')


def require_dict(manifest, field_name, manifest_name):
    value = manifest.get(field_name)
    if not isinstance(value, dict) or not value:
        raise ValueError(f'{manifest_name} is missing required dict field: {field_name}')
    return value


def require_field(manifest, field_name, manifest_name):
    value = manifest.get(field_name)
    if value in {None, ''}:
        raise ValueError(f'{manifest_name} is missing required field: {field_name}')
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


def build_readme_block(single_manifest=None, multi_manifest=None):
    lines = [
        README_START,
        '### Generated Benchmark Snapshot',
        '',
        'This section is generated from the official component benchmark manifests in `data/outputs/`.',
        'The published component-model scores come from the untouched `2026` holdout.',
        ''
    ]
    lines.extend(build_single_lines(single_manifest))
    lines.extend(build_multi_lines(multi_manifest))
    lines.append(README_END)
    return '\n'.join(lines)


def update_component_readme(single_manifest_path=None, multi_manifest_path=None, readme_path=None, write_summary=True):
    readme_path = Path(readme_path) if readme_path is not None else PROJECT_ROOT / 'README.md'
    text = readme_path.read_text(encoding='utf-8')

    single_path = Path(single_manifest_path) if single_manifest_path is not None else OUTPUTS_DIR / COMPONENT_SINGLE_OFFICIAL_MANIFEST
    multi_path = Path(multi_manifest_path) if multi_manifest_path is not None else OUTPUTS_DIR / COMPONENT_MULTI_OFFICIAL_MANIFEST

    single_manifest = validate_single_manifest(load_manifest(single_path))
    multi_manifest = validate_multi_manifest(load_manifest(multi_path))
    block = build_readme_block(single_manifest=single_manifest, multi_manifest=multi_manifest)

    if README_START not in text or README_END not in text:
        raise ValueError('README benchmark markers not found')

    start_idx = text.index(README_START)
    end_idx = text.index(README_END) + len(README_END)
    updated = text[:start_idx] + block + text[end_idx:]
    readme_path.write_text(updated, encoding='utf-8')
    if write_summary:
        write_summary_artifacts(single_manifest=single_manifest, multi_manifest=multi_manifest)
    return readme_path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Update the README component benchmark section from the official manifests'
    )
    parser.add_argument('--single-manifest', default=None)
    parser.add_argument('--multi-manifest', default=None)
    parser.add_argument('--readme-path', default=None)
    parser.add_argument('--summary', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    readme_path = update_component_readme(
        single_manifest_path=args.single_manifest,
        multi_manifest_path=args.multi_manifest,
        readme_path=args.readme_path,
        write_summary=args.summary
    )
    print(f'[write] {readme_path}')
    print('[done] README benchmark section refreshed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
