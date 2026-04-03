import argparse
import json
from pathlib import Path

from src.config.paths import OUTPUTS_DIR, PROJECT_ROOT

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
README_START = '<!-- COMPONENT_BENCHMARK_START -->'
README_END = '<!-- COMPONENT_BENCHMARK_END -->'
DEFAULT_SINGLE_MANIFEST = OUTPUTS_DIR / 'component_single_label_benchmark_manifest.json'
DEFAULT_MULTI_MANIFEST = OUTPUTS_DIR / 'component_multilabel_manifest.json'


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


def build_readme_block(single_manifest=None, multi_manifest=None):
    lines = [
        README_START,
        '### Generated Benchmark Snapshot',
        '',
        'This section is generated from the benchmark manifests in `data/outputs/`.',
        'The official published component-model scores come from the untouched `2026` holdout.',
        ''
    ]

    if single_manifest is not None:
        holdout = single_manifest.get('official_holdout_metrics', {})
        lines.extend(
            [
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
        )
    else:
        lines.extend(
            [
                '#### Single-label structured benchmark',
                '',
                '- No final single-label benchmark manifest found yet',
                ''
            ]
        )

    if multi_manifest is not None:
        holdout = multi_manifest.get('official_holdout_metrics', {})
        lines.extend(
            [
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
        )
    else:
        lines.extend(
            [
                '#### Multi-label routing benchmark',
                '',
                '- No final multi-label benchmark manifest found yet',
                ''
            ]
        )

    lines.append(README_END)
    return '\n'.join(lines)


def update_component_readme(single_manifest_path=None, multi_manifest_path=None, readme_path=None):
    readme_path = Path(readme_path) if readme_path is not None else PROJECT_ROOT / 'README.md'
    text = readme_path.read_text(encoding='utf-8')
    single_manifest = load_manifest(single_manifest_path or DEFAULT_SINGLE_MANIFEST)
    multi_manifest = load_manifest(multi_manifest_path or DEFAULT_MULTI_MANIFEST)
    block = build_readme_block(single_manifest=single_manifest, multi_manifest=multi_manifest)

    if README_START not in text or README_END not in text:
        raise ValueError('README benchmark markers not found')

    start_idx = text.index(README_START)
    end_idx = text.index(README_END) + len(README_END)
    updated = text[:start_idx] + block + text[end_idx:]
    readme_path.write_text(updated, encoding='utf-8')
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
    return parser.parse_args()


def main():
    args = parse_args()
    readme_path = update_component_readme(
        single_manifest_path=args.single_manifest,
        multi_manifest_path=args.multi_manifest,
        readme_path=args.readme_path
    )
    print(f'[write] {readme_path}')
    print('[done] README benchmark section refreshed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
