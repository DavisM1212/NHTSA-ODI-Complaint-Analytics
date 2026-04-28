import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config.contracts import (
    SEVERITY_URGENCY_OFFICIAL_CALIBRATION,
    SEVERITY_URGENCY_OFFICIAL_MANIFEST,
    SEVERITY_URGENCY_OFFICIAL_METRICS,
    SEVERITY_URGENCY_OFFICIAL_REVIEW_BUDGETS,
)
from src.config.paths import OUTPUTS_DIR, PROJECT_ROOT

# -----------------------------------------------------------------------------
# Artifact names
# -----------------------------------------------------------------------------
DEFAULT_FIGURE_DIR = PROJECT_ROOT / 'docs' / 'figures' / 'severity_model'
OFFICIAL_MANIFEST = SEVERITY_URGENCY_OFFICIAL_MANIFEST
OFFICIAL_METRICS = SEVERITY_URGENCY_OFFICIAL_METRICS
OFFICIAL_REVIEW_BUDGETS = SEVERITY_URGENCY_OFFICIAL_REVIEW_BUDGETS
OFFICIAL_CALIBRATION = SEVERITY_URGENCY_OFFICIAL_CALIBRATION

FIGURE_INDEX = 'severity_model_figure_index.csv'

FIGURE_STYLE = {
    'official': '#1f77b4',
    'baseline': '#9aa0a6',
    'raw': '#d62728',
    'sigmoid': '#2ca02c',
    'isotonic': '#ff7f0e',
    'accent': '#1b9e77',
    'grid': '#e8e8e8'
}

CALIBRATION_COLORS = {
    'late_fusion_raw': FIGURE_STYLE['raw'],
    'late_fusion_sigmoid': FIGURE_STYLE['sigmoid'],
    'late_fusion_isotonic': FIGURE_STYLE['isotonic']
}

SPLIT_LABELS = {
    'train': 'Train Through 2024',
    'valid_2025': 'Validation 2025',
    'holdout_2026': 'Holdout 2026'
}


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def read_csv(outputs_dir, name):
    path = Path(outputs_dir) / name
    if not path.exists():
        raise FileNotFoundError(f'Missing required severity artifact: {path}')
    return pd.read_csv(path)


def read_json(outputs_dir, name):
    path = Path(outputs_dir) / name
    if not path.exists():
        raise FileNotFoundError(f'Missing required severity artifact: {path}')
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def figure_path(output_dir, name):
    return Path(output_dir) / f'{name}.png'


def display_path(path):
    path = Path(path)
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def coerce_bool_series(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({'true', '1', 'yes', 'y'})


def save_figure(fig, output_dir, name, title, description):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = figure_path(output_dir, name)
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return {
        'figure': name,
        'path': display_path(path),
        'title': title,
        'description': description
    }


def setup_axes(ax, title, xlabel=None, ylabel=None, grid_axis='y'):
    ax.set_title(title, loc='left', fontsize=14, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis=grid_axis, color=FIGURE_STYLE['grid'], linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def format_pct(value):
    return f'{100 * float(value):.1f}%'


def write_index(rows, output_dir):
    output_dir = Path(output_dir)
    index_path = output_dir / FIGURE_INDEX
    pd.DataFrame(rows).to_csv(index_path, index=False)
    return index_path


def get_metric_row(metrics_df, model_name, split_name):
    row = metrics_df.loc[
        metrics_df['model'].eq(model_name) & metrics_df['split'].eq(split_name)
    ]
    if row.empty:
        raise KeyError(f'Missing metrics row for {model_name} on {split_name}')
    return row.iloc[0]


def get_budget_rows(review_budget_df, model_name, split_name):
    subset = review_budget_df.loc[
        review_budget_df['model'].eq(model_name) & review_budget_df['split'].eq(split_name)
    ].copy()
    if subset.empty:
        raise KeyError(f'Missing review budget rows for {model_name} on {split_name}')
    return subset.sort_values('budget_fraction').reset_index(drop=True)


def get_calibration_rows(calibration_df, model_name, split_name):
    subset = calibration_df.loc[
        calibration_df['model'].eq(model_name) & calibration_df['split'].eq(split_name)
    ].copy()
    if subset.empty:
        raise KeyError(f'Missing calibration rows for {model_name} on {split_name}')
    return subset.sort_values('bin').reset_index(drop=True)


def metric_label(model_name):
    return {
        'dummy_prior': 'Dummy prior',
        'late_fusion_raw': 'Late fusion raw',
        'late_fusion_sigmoid': 'Late fusion + sigmoid',
        'late_fusion_isotonic': 'Late fusion + isotonic'
    }.get(model_name, str(model_name))


# -----------------------------------------------------------------------------
# Figure builders
# -----------------------------------------------------------------------------
def plot_split_context(manifest, output_dir):
    split_df = pd.DataFrame(manifest['split_summary']).copy()
    split_df['label'] = split_df['split'].map(SPLIT_LABELS).fillna(split_df['split'])
    split_df = split_df.set_index('split').loc[['train', 'valid_2025', 'holdout_2026']].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    positions = np.arange(len(split_df))

    axes[0].barh(positions, split_df['rows'].astype(float), color=FIGURE_STYLE['official'])
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(split_df['label'])
    axes[0].invert_yaxis()
    for idx, value in enumerate(split_df['rows'].astype(int)):
        axes[0].text(value, idx, f' {value:,}', va='center', fontsize=9)
    setup_axes(axes[0], 'Rows By Benchmark Split', xlabel='Rows', grid_axis='x')

    axes[1].barh(positions, split_df['positive_rate'].astype(float), color=FIGURE_STYLE['accent'])
    axes[1].set_yticks(positions)
    axes[1].set_yticklabels(split_df['label'])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, max(0.10, float(split_df['positive_rate'].max()) * 1.15))
    for idx, value in enumerate(split_df['positive_rate'].astype(float)):
        axes[1].text(value, idx, f' {format_pct(value)}', va='center', fontsize=9)
    setup_axes(axes[1], 'Severe-Case Rate By Benchmark Split', xlabel='Positive rate', grid_axis='x')

    fig.suptitle('Severity Benchmark Data Window', fontsize=16, fontweight='bold', x=0.02, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return save_figure(
        fig,
        output_dir,
        'severity_split_context',
        'Severity Benchmark Data Window',
        'Row counts and severe-case rates for the train, validation, and holdout severity benchmark splits'
    )


def plot_official_vs_baseline(metrics_df, manifest, output_dir):
    baseline_name = manifest['baseline_model_name']
    official_name = manifest['official_model_name']
    metrics = [
        ('pr_auc', 'PR-AUC'),
        ('recall_top_5pct', 'Recall @ Top 5%'),
        ('precision_top_5pct', 'Precision @ Top 5%'),
        ('recall_top_10pct', 'Recall @ Top 10%')
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)

    for ax, split_name in zip(axes, ['valid_2025', 'holdout_2026']):
        baseline_row = get_metric_row(metrics_df, baseline_name, split_name)
        official_row = get_metric_row(metrics_df, official_name, split_name)
        positions = np.arange(len(metrics))
        width = 0.36

        baseline_values = [float(baseline_row[field]) for field, _ in metrics]
        official_values = [float(official_row[field]) for field, _ in metrics]

        ax.bar(
            positions - width / 2,
            baseline_values,
            width,
            color=FIGURE_STYLE['baseline'],
            label='Dummy prior'
        )
        ax.bar(
            positions + width / 2,
            official_values,
            width,
            color=FIGURE_STYLE['official'],
            label='Official severity model'
        )

        ax.set_xticks(positions)
        ax.set_xticklabels([label for _, label in metrics], rotation=20, ha='right')
        ax.set_ylim(0, 1.02)
        setup_axes(ax, SPLIT_LABELS.get(split_name, split_name), ylabel='Metric value' if split_name == 'valid_2025' else None)
        ax.text(
            0.34,
            0.965,
            f"Brier\nOfficial {float(official_row['brier_score']):.4f}\nBaseline {float(baseline_row['brier_score']):.4f}",
            transform=ax.transAxes,
            fontsize=9,
            ha='center',
            va='top',
            bbox={'boxstyle': 'round,pad=0.35', 'facecolor': 'white', 'edgecolor': '#cccccc'}
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2)
    fig.suptitle('Official Severity Model Vs Baseline', fontsize=16, fontweight='bold', x=0.02, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return save_figure(
        fig,
        output_dir,
        'severity_official_vs_baseline',
        'Official Severity Model Vs Baseline',
        'Validation and holdout comparison between the dummy prior baseline and the official calibrated late-fusion severity model'
    )


def plot_review_budget_tradeoff(review_budget_df, manifest, output_dir):
    baseline_name = manifest['baseline_model_name']
    official_name = manifest['official_model_name']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)

    for ax, split_name in zip(axes, ['valid_2025', 'holdout_2026']):
        baseline_rows = get_budget_rows(review_budget_df, baseline_name, split_name)
        official_rows = get_budget_rows(review_budget_df, official_name, split_name)
        x = baseline_rows['budget_fraction'].astype(float).to_numpy() * 100

        ax.plot(x, official_rows['recall_within_flagged_set'].astype(float), marker='o', linewidth=2.3, color=FIGURE_STYLE['official'], label='Official recall')
        ax.plot(x, official_rows['precision_within_flagged_set'].astype(float), marker='o', linewidth=2.3, color=FIGURE_STYLE['accent'], label='Official precision')
        ax.plot(x, baseline_rows['recall_within_flagged_set'].astype(float), marker='x', linestyle='--', linewidth=1.8, color=FIGURE_STYLE['baseline'], label='Baseline recall')
        ax.plot(x, baseline_rows['precision_within_flagged_set'].astype(float), marker='x', linestyle='--', linewidth=1.8, color='#c7c7c7', label='Baseline precision')

        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(v)}%' for v in x])
        ax.set_ylim(0, 1.02)
        setup_axes(ax, SPLIT_LABELS.get(split_name, split_name), xlabel='Review budget', ylabel='Metric value' if split_name == 'valid_2025' else None)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=4)
    fig.suptitle('Review Budget Tradeoff', fontsize=16, fontweight='bold', x=0.02, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return save_figure(
        fig,
        output_dir,
        'severity_review_budget_tradeoff',
        'Review Budget Tradeoff',
        'How recall and precision change as reviewers inspect the top 1%, 2%, 5%, or 10% of complaints'
    )


def plot_captured_cases(review_budget_df, manifest, output_dir):
    baseline_name = manifest['baseline_model_name']
    official_name = manifest['official_model_name']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    for ax, split_name in zip(axes, ['valid_2025', 'holdout_2026']):
        baseline_rows = get_budget_rows(review_budget_df, baseline_name, split_name)
        official_rows = get_budget_rows(review_budget_df, official_name, split_name)
        labels = [label.replace('top_', '').replace('pct', '%') for label in official_rows['budget_label']]
        positions = np.arange(len(labels))
        width = 0.36

        baseline_values = baseline_rows['severe_cases_captured'].astype(float).to_numpy()
        official_values = official_rows['severe_cases_captured'].astype(float).to_numpy()

        ax.bar(positions - width / 2, baseline_values, width, color=FIGURE_STYLE['baseline'], label='Dummy prior')
        ax.bar(positions + width / 2, official_values, width, color=FIGURE_STYLE['official'], label='Official severity model')

        for idx, value in enumerate(official_values):
            gain = int(round(value - baseline_values[idx]))
            ax.text(idx + width / 2, value, f' +{gain}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        setup_axes(ax, SPLIT_LABELS.get(split_name, split_name), xlabel='Review budget', ylabel='Severe complaints captured' if split_name == 'valid_2025' else None)

    axes[0].legend(frameon=False, loc='upper left')
    fig.suptitle('Captured Severe Complaints By Review Budget', fontsize=16, fontweight='bold', x=0.02, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return save_figure(
        fig,
        output_dir,
        'severity_captured_cases_by_budget',
        'Captured Severe Complaints By Review Budget',
        'How many severe complaints the official model captures at each review budget compared with the dummy baseline'
    )


def plot_calibration(calibration_df, metrics_df, model_names, split_name, output_dir, figure_name, title, description):
    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    ax.plot([0, 1], [0, 1], linestyle='--', color='#555555', linewidth=1)

    for model_name in model_names:
        model_rows = get_calibration_rows(calibration_df, model_name, split_name)
        metric_row = get_metric_row(metrics_df, model_name, split_name)
        color = CALIBRATION_COLORS.get(model_name, FIGURE_STYLE['official'])
        label = f"{metric_label(model_name)} (Brier {float(metric_row['brier_score']):.4f})"
        ax.plot(
            model_rows['avg_score'].astype(float),
            model_rows['observed_rate'].astype(float),
            marker='o',
            linewidth=2,
            color=color,
            label=label
        )

    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    setup_axes(
        ax,
        title,
        xlabel='Average predicted score',
        ylabel='Observed severe-case rate',
        grid_axis='both'
    )
    ax.legend(frameon=False, loc='upper left')
    return save_figure(fig, output_dir, figure_name, title, description)


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def generate_severity_visuals(outputs_dir=OUTPUTS_DIR, output_dir=DEFAULT_FIGURE_DIR):
    outputs_dir = Path(outputs_dir)
    output_dir = Path(output_dir)

    manifest = read_json(outputs_dir, OFFICIAL_MANIFEST)
    metrics_df = read_csv(outputs_dir, OFFICIAL_METRICS)
    review_budget_df = read_csv(outputs_dir, OFFICIAL_REVIEW_BUDGETS)
    calibration_df = read_csv(outputs_dir, OFFICIAL_CALIBRATION)

    metrics_df['is_official'] = coerce_bool_series(metrics_df['is_official'])
    metrics_df['is_baseline'] = coerce_bool_series(metrics_df['is_baseline'])
    review_budget_df['is_official'] = coerce_bool_series(review_budget_df['is_official'])
    review_budget_df['is_baseline'] = coerce_bool_series(review_budget_df['is_baseline'])
    calibration_df['is_official'] = coerce_bool_series(calibration_df['is_official'])

    official_name = str(manifest['official_model_name'])
    validation_models = ['late_fusion_raw', 'late_fusion_sigmoid', 'late_fusion_isotonic']
    validation_models = [model_name for model_name in validation_models if model_name in set(calibration_df['model'])]

    holdout_models = ['late_fusion_raw']
    if official_name not in holdout_models:
        holdout_models.append(official_name)

    rows = [
        plot_split_context(manifest, output_dir),
        plot_official_vs_baseline(metrics_df, manifest, output_dir),
        plot_review_budget_tradeoff(review_budget_df, manifest, output_dir),
        plot_captured_cases(review_budget_df, manifest, output_dir),
        plot_calibration(
            calibration_df,
            metrics_df,
            validation_models,
            'valid_2025',
            output_dir,
            'severity_validation_calibration',
            'Validation Calibration Comparison',
            'Reliability curves for the raw, sigmoid-calibrated, and isotonic-calibrated severity candidates on validation'
        ),
        plot_calibration(
            calibration_df,
            metrics_df,
            holdout_models,
            'holdout_2026',
            output_dir,
            'severity_holdout_calibration',
            'Holdout Calibration Check',
            'Reliability comparison between the raw late-fusion score and the promoted official severity model on the 2026 holdout'
        )
    ]
    index_path = write_index(rows, output_dir)
    return {
        'output_dir': output_dir,
        'index_path': index_path,
        'figures': rows
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate presentation-ready severity model figures from official severity artifacts'
    )
    parser.add_argument('--outputs-dir', default=str(OUTPUTS_DIR))
    parser.add_argument('--output-dir', default=str(DEFAULT_FIGURE_DIR))
    return parser.parse_args()


def main():
    args = parse_args()
    result = generate_severity_visuals(
        outputs_dir=args.outputs_dir,
        output_dir=args.output_dir
    )

    print(f"[write] {result['output_dir']}")
    for row in result['figures']:
        print(f"[figure] {row['path']}")
    print(f"[index] {result['index_path']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
