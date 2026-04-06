import argparse
import json
from pathlib import Path

from src.config.paths import OUTPUTS_DIR, PROJECT_ROOT

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
README_START = '<!-- COMPONENT_BENCHMARK_START -->'
README_END = '<!-- COMPONENT_BENCHMARK_END -->'
DEFAULT_SINGLE_MANIFEST = OUTPUTS_DIR / 'component_textwave2b_calibration_manifest.json'
FALLBACK_SINGLE_MANIFEST = OUTPUTS_DIR / 'component_single_label_benchmark_manifest.json'
DEFAULT_MULTI_MANIFEST = OUTPUTS_DIR / 'component_multilabel_manifest.json'
SUMMARY_CSV = OUTPUTS_DIR / 'component_official_benchmark_summary.csv'
SUMMARY_JSON = OUTPUTS_DIR / 'component_official_benchmark_summary.json'


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_manifest(manifest_path):
    if manifest_path is None:
        return None

    path = Path(manifest_path)
    if not path.exists():
        return None

    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def format_metric(value):
    if value is None:
        return 'n/a'
    try:
        return f'{float(value):.4f}'
    except Exception:
        return str(value)


def default_single_manifest_path():
    if DEFAULT_SINGLE_MANIFEST.exists():
        return DEFAULT_SINGLE_MANIFEST
    return FALLBACK_SINGLE_MANIFEST


def single_manifest_kind(single_manifest):
    if single_manifest is None:
        return None
    if single_manifest.get('artifact_role') == 'text_wave2b_calibration':
        return 'text_wave2b_calibration'
    return 'structured_catboost'


def single_holdout_metrics(single_manifest):
    if single_manifest_kind(single_manifest) == 'text_wave2b_calibration':
        return single_manifest.get('calibrated_holdout_metrics', {})
    return single_manifest.get('official_holdout_metrics', {})


def build_single_lines(single_manifest):
    if single_manifest is None:
        return [
            '#### Single-label component benchmark',
            '',
            '- No final single-label benchmark manifest found yet',
            ''
        ]

    kind = single_manifest_kind(single_manifest)
    holdout = single_holdout_metrics(single_manifest)

    if kind == 'text_wave2b_calibration':
        lines = [
            '#### Single-label component benchmark',
            '',
            '- Scope: scoped benchmark on the single-label complaint subset',
            f"- Model: `{single_manifest.get('family_name', holdout.get('model', 'n/a'))}`",
            f"- Inputs: complaint narrative text + `{single_manifest.get('structured_feature_set', 'n/a')}` structured companion features",
            f"- Text weight: `{single_manifest.get('text_weight', 'n/a')}`",
            f"- Final text model: `{single_manifest.get('final_linear_model', 'n/a')}`",
            f"- Calibration: `{single_manifest.get('calibration_method', 'n/a')}` alpha `{single_manifest.get('selected_alpha', 'n/a')}` from `{single_manifest.get('calibration_source', 'n/a')}`",
            f"- Structured branch iteration: `{single_manifest.get('selected_iteration', holdout.get('selected_iteration', 'n/a'))}`",
            f"- Holdout macro F1: `{format_metric(holdout.get('macro_f1'))}`",
            f"- Holdout top-1 accuracy: `{format_metric(holdout.get('top_1_accuracy'))}`",
            f"- Holdout top-3 accuracy: `{format_metric(holdout.get('top_3_accuracy'))}`",
            f"- Holdout calibration ECE: `{format_metric(single_manifest.get('holdout_ece'))}`",
            ''
        ]
        return lines

    return [
        '#### Single-label structured benchmark',
        '',
        '- Scope: scoped baseline on the single-label complaint subset',
        f"- Feature set: `{single_manifest.get('selected_feature_set', 'n/a')}`",
        f"- Selected CatBoost iteration: `{single_manifest.get('selected_iteration', 'n/a')}`",
        f"- Holdout macro F1: `{format_metric(holdout.get('macro_f1'))}`",
        f"- Holdout top-1 accuracy: `{format_metric(holdout.get('top_1_accuracy'))}`",
        f"- Holdout top-3 accuracy: `{format_metric(holdout.get('top_3_accuracy'))}`",
        ''
    ]


def build_multi_lines(multi_manifest):
    if multi_manifest is None:
        return [
            '#### Multi-label routing benchmark',
            '',
            '- No final multi-label benchmark manifest found yet',
            ''
        ]

    holdout = multi_manifest.get('official_holdout_metrics', {})
    return [
        '#### Multi-label routing benchmark',
        '',
        '- Scope: full kept-case complaint routing benchmark',
        f"- Model: `{multi_manifest.get('selected_model', holdout.get('model', 'n/a'))}`",
        f"- Feature set: `{multi_manifest.get('selected_feature_set', 'n/a')}`",
        f"- Threshold: `{multi_manifest.get('selected_threshold', 'n/a')}`",
        f"- Selected iteration: `{multi_manifest.get('selected_iteration', 'n/a')}`",
        f"- Holdout macro F1: `{format_metric(holdout.get('macro_f1'))}`",
        f"- Holdout micro F1: `{format_metric(holdout.get('micro_f1'))}`",
        f"- Holdout recall@3: `{format_metric(holdout.get('recall_at_3'))}`",
        f"- Holdout precision@3: `{format_metric(holdout.get('precision_at_3'))}`",
        ''
    ]


def build_summary_rows(single_manifest=None, multi_manifest=None):
    rows = []

    if single_manifest is not None:
        holdout = single_holdout_metrics(single_manifest)
        rows.append(
            {
                'task': 'single_label_component',
                'official_model': single_manifest.get('family_name', holdout.get('model', 'n/a')),
                'artifact_role': single_manifest.get('artifact_role', 'final_benchmark'),
                'feature_set': single_manifest.get('structured_feature_set', single_manifest.get('selected_feature_set', 'n/a')),
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
        holdout = multi_manifest.get('official_holdout_metrics', {})
        rows.append(
            {
                'task': 'multi_label_component_routing',
                'official_model': multi_manifest.get('selected_model', holdout.get('model', 'n/a')),
                'artifact_role': multi_manifest.get('artifact_role', 'final_benchmark'),
                'feature_set': multi_manifest.get('selected_feature_set', 'n/a'),
                'holdout_rows': holdout.get('rows'),
                'macro_f1': holdout.get('macro_f1'),
                'micro_f1': holdout.get('micro_f1'),
                'recall_at_3': holdout.get('recall_at_3'),
                'precision_at_3': holdout.get('precision_at_3'),
                'label_coverage': holdout.get('label_coverage'),
                'threshold': holdout.get('threshold', multi_manifest.get('selected_threshold')),
                'selected_iteration': holdout.get('selected_iteration', multi_manifest.get('selected_iteration')),
                'status': 'official'
            }
        )

    return rows


def write_summary_artifacts(single_manifest=None, multi_manifest=None, summary_csv_path=None, summary_json_path=None):
    rows = build_summary_rows(single_manifest=single_manifest, multi_manifest=multi_manifest)
    csv_path = Path(summary_csv_path) if summary_csv_path is not None else SUMMARY_CSV
    json_path = Path(summary_json_path) if summary_json_path is not None else SUMMARY_JSON

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
        'This section is generated from the benchmark manifests in `data/outputs/`.',
        'The official published component-model scores come from the untouched `2026` holdout.',
        ''
    ]

    lines.extend(build_single_lines(single_manifest))
    lines.extend(build_multi_lines(multi_manifest))

    lines.append(README_END)
    return '\n'.join(lines)


def update_component_readme(single_manifest_path=None, multi_manifest_path=None, readme_path=None, write_summary=True):
    readme_path = Path(readme_path) if readme_path is not None else PROJECT_ROOT / 'README.md'
    text = readme_path.read_text(encoding='utf-8')
    single_manifest = load_manifest(single_manifest_path or default_single_manifest_path())
    multi_manifest = load_manifest(multi_manifest_path or DEFAULT_MULTI_MANIFEST)
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


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description='Update the README component benchmark section from benchmark manifests'
    )
    parser.add_argument('--single-manifest', default=None)
    parser.add_argument('--multi-manifest', default=None)
    parser.add_argument('--readme-path', default=None)
    parser.add_argument('--no-summary', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    readme_path = update_component_readme(
        single_manifest_path=args.single_manifest,
        multi_manifest_path=args.multi_manifest,
        readme_path=args.readme_path,
        write_summary=not args.no_summary
    )
    print(f'[write] {readme_path}')
    print('[done] README benchmark section refreshed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
